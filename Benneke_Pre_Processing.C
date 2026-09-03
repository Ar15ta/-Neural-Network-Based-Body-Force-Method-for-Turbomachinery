/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "clockTime.H"
#include "IFstream.H"
#include "OFstream.H"
#include "fvMesh.H"
#include "volFields.H"
#include "cellSet.H"
#include "vectorField.H"
#include "scalarField.H"
#include "ListListOps.H"
#include "PtrList.H"
#include "token.H"
#include "treeBoundBox.H"
#include "SquareMatrix.H"
#include "LUscalarMatrix.H"
#include "SVD.H"
#include "ListOps.H"
#include <sstream>

using namespace Foam;

// 通量场参数结构
struct FluxFieldParams {
    scalar epsilon;
    label neighbors;
    label polyDegree;
};

// 读取优化参数
void readFluxOptimizationParameters(
    const fileName& paramFileName,
    FluxFieldParams& qrtParams,
    FluxFieldParams& qztParams,
    FluxFieldParams& qrzParams,
    FluxFieldParams& qzzParams,
    FluxFieldParams& qrrParams,
    FluxFieldParams& qttParams)
{
    IFstream paramFile(paramFileName);
    if (!paramFile.good())
    {
        FatalErrorInFunction << "Cannot open file " << paramFileName << exit(FatalError);
    }
    
    string line;
    FluxFieldParams* currentParams = nullptr;
    
    while (paramFile.getLine(line))
    {
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();
        
        if (line.find("Q_RT FIELD:") != string::npos) {
            currentParams = &qrtParams;
        } else if (line.find("Q_ZT FIELD:") != string::npos) {
            currentParams = &qztParams;
        } else if (line.find("Q_RZ FIELD:") != string::npos) {
            currentParams = &qrzParams;
        } else if (line.find("Q_ZZ FIELD:") != string::npos) {
            currentParams = &qzzParams;
        } else if (line.find("Q_RR FIELD:") != string::npos) {
            currentParams = &qrrParams;
        } else if (line.find("Q_TT FIELD:") != string::npos) {
            currentParams = &qttParams;
        } else if (currentParams && line.find("Epsilon:") != string::npos) {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos) {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> currentParams->epsilon;
            }
        } else if (currentParams && line.find("Neighbors:") != string::npos) {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos) {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> currentParams->neighbors;
            }
        } else if (currentParams && line.find("Polynomial Degree:") != string::npos) {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos) {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> currentParams->polyDegree;
            }
        }
    }
    
    Info << "读取优化参数完成:" << endl;
    Info << "  Q_RT: epsilon=" << qrtParams.epsilon << ", neighbors=" << qrtParams.neighbors << ", polyDegree=" << qrtParams.polyDegree << endl;
    Info << "  Q_ZT: epsilon=" << qztParams.epsilon << ", neighbors=" << qztParams.neighbors << ", polyDegree=" << qztParams.polyDegree << endl;
    Info << "  Q_RZ: epsilon=" << qrzParams.epsilon << ", neighbors=" << qrzParams.neighbors << ", polyDegree=" << qrzParams.polyDegree << endl;
    Info << "  Q_ZZ: epsilon=" << qzzParams.epsilon << ", neighbors=" << qzzParams.neighbors << ", polyDegree=" << qzzParams.polyDegree << endl;
    Info << "  Q_RR: epsilon=" << qrrParams.epsilon << ", neighbors=" << qrrParams.neighbors << ", polyDegree=" << qrrParams.polyDegree << endl;
    Info << "  Q_TT: epsilon=" << qttParams.epsilon << ", neighbors=" << qttParams.neighbors << ", polyDegree=" << qttParams.polyDegree << endl;
}

// 通量数据结构
struct FluxData {
    scalar r;
    scalar z;
    scalar rho;
    scalar trr;  // rho*Vr^2 + p
    scalar trt;  // rho*Vr*Vtheta
    scalar trz;  // rho*Vr*Vz
    scalar ttt;  // rho*Vtheta^2 + p
    scalar tzt;  // rho*Vz*Vtheta
    scalar tzz;  // rho*Vz^2 + p
    scalar lambda;
};

// 读取通量数据
void readFluxData(const fileName& inputFileName, List<FluxData>& fluxData, vectorField& points, 
                  scalarField& qrt, scalarField& qzt, scalarField& qrz, 
                  scalarField& qzz, scalarField& qrr, scalarField& qtt, scalarField& rhoField, scalarField& lambdaField)
{
    IFstream file(inputFileName);
    if (!file.good())
    {
        FatalErrorInFunction << "Cannot open file " << inputFileName << exit(FatalError);
    }

    Info << "正在读取文件: " << inputFileName << endl;

    string line;
    label lineNumber = 0;
    
    // 读取数据
    while (file.getLine(line))
    {
        lineNumber++;
        if (line.empty()) break;
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();

        // 跳过表头
        if (line.find("R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda") != string::npos)
            continue;

        std::stringstream ss(line);
        std::string token;
        scalar values[10]; // openFOAM_Input_Force.csv有10列
        label col = 0;

        while (std::getline(ss, token, ',') && col < 10)
        {
            try { values[col] = std::stod(token); } catch (...) { values[col] = 0; }
            col++;
        }

        if (col >= 10 && values[0] > 0)
        {
            FluxData data;
            data.r = values[0];
            data.rho = values[1];
            data.trr = values[2];
            data.trt = values[3];
            data.trz = values[4];
            data.ttt = values[5];
            data.tzt = values[6];
            data.tzz = values[7];
            data.z = values[8];
            data.lambda = values[9]; // 直接从CSV文件读取lambda值
            fluxData.append(data);
        }
    }

    label nPoints = fluxData.size();
    Info << "读取到 " << nPoints << " 个数据点" << endl;
    
    if (nPoints == 0)
    {
        FatalErrorInFunction << "文件中没有找到有效数据点" << exit(FatalError);
    }
    
    // 初始化字段
    points.setSize(nPoints);
    qrt.setSize(nPoints);
    qzt.setSize(nPoints);
    qrz.setSize(nPoints);
    qzz.setSize(nPoints);
    qrr.setSize(nPoints);
    qtt.setSize(nPoints);
    rhoField.setSize(nPoints);
    lambdaField.setSize(nPoints);
    
    // 计算通量场
    forAll(fluxData, i)
    {
        const FluxData& data = fluxData[i];
        points[i] = vector(data.r, 0.0, data.z);
        
        // 使用CSV文件中的lambda值计算通量场
        qrt[i] = data.r * data.r * data.lambda * data.trt;
        qzt[i] = data.r * data.r * data.lambda * data.tzt;
        qrz[i] = data.r * data.lambda * data.trz;
        qzz[i] = data.r * data.lambda * data.tzz;
        qrr[i] = data.r * data.lambda * data.trr;
        qtt[i] = data.lambda * data.ttt;
        
        // 存储rho和lambda值
        rhoField[i] = data.rho;
        lambdaField[i] = data.lambda;
    }

    // 输出lambda范围
    scalar minLambda = GREAT;
    scalar maxLambda = -GREAT;
    forAll(lambdaField, i)
    {
        minLambda = Foam::min(minLambda, lambdaField[i]);
        maxLambda = Foam::max(maxLambda, lambdaField[i]);
    }
    Info << "Lambda范围: " << minLambda << " - " << maxLambda << endl;
}

// 查找最近的N个邻居点
void findNearestNeighbors(
    const vector& target,
    const vectorField& points,
    labelList& neighborIndices,
    scalarField& neighborDistances)
{
    label nPoints = points.size();
    label nNeighbors = neighborIndices.size();
    
    neighborDistances.setSize(nNeighbors, GREAT);
    neighborIndices.setSize(nNeighbors, -1);
    
    for (label i = 0; i < nPoints; i++)
    {
        scalar dist = mag(points[i] - target);
        
        for (label j = 0; j < nNeighbors; j++)
        {
            if (dist < neighborDistances[j])
            {
                for (label k = nNeighbors - 1; k > j; k--)
                {
                    neighborDistances[k] = neighborDistances[k-1];
                    neighborIndices[k] = neighborIndices[k-1];
                }
                neighborDistances[j] = dist;
                neighborIndices[j] = i;
                break;
            }
        }
    }
}

// IDW向量插值器 - 用于插值梯度向量场
class IDWVectorInterpolator
{
private:
    const vectorField& points_;
    const vectorField& values_;
    label nPoints_;
    label nNeighbors_;
    scalar power_;

public:
    IDWVectorInterpolator(
        const vectorField& points,
        const vectorField& values,
        label nNeighbors = 12,
        scalar power = 2.0)
        : points_(points),
          values_(values),
          nPoints_(points.size()),
          nNeighbors_(nNeighbors),
          power_(power)
    {}

    vector interpolate(const vector& target) const
    {
        label nNeighbors = min(nPoints_, nNeighbors_);
        
        labelList neighborIndices(nNeighbors);
        scalarField neighborDistances;
        findNearestNeighbors(target, points_, neighborIndices, neighborDistances);
        
        scalar sumWeight = 0.0;
        vector sumWeightedValue(0, 0, 0);
        
        for (label i = 0; i < nNeighbors; i++)
        {
            label ptI = neighborIndices[i];
            scalar dist = neighborDistances[i];
            
            if (dist < 1e-10)
            {
                return values_[ptI];
            }
            
            scalar weight = 1.0 / Foam::pow(dist, power_);
            sumWeight += weight;
            sumWeightedValue += weight * values_[ptI];
        }
        
        if (sumWeight < 1e-10)
        {
            return values_[neighborIndices[0]];
        }
        
        return sumWeightedValue / sumWeight;
    }
};

// RBF插值器类
class RBFInterpolator
{
private:
    const vectorField& points_;
    const scalarField& values_;
    scalar eps_;
    label nPoints_;
    label nNeighbors_;
    label polyDegree_;
    label polyDim_;
    
    label getPolyDim() const
    {
        if (polyDegree_ == 0) return 1;
        if (polyDegree_ == 1) return 3;
        if (polyDegree_ == 2) return 6;
        return 1;
    }
    
    void polynomialBasis(const vector& p, scalarField& basis) const
    {
        scalar x = p.x();
        scalar y = p.z();
        
        basis.setSize(polyDim_);
        
        if (polyDegree_ == 0)
        {
            basis[0] = 1.0;
        }
        else if (polyDegree_ == 1)
        {
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
        }
        else if (polyDegree_ == 2)
        {
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * x;
            basis[4] = x * y;
            basis[5] = y * y;
        }
    }

    scalar distance(const vector& p1, const vector& p2) const
    {
        return mag(p1 - p2);
    }

    scalar gaussianKernel(const vector& p, const vector& p0) const
    {
        scalar d = distance(p, p0);
        return Foam::exp(-pow(eps_ * d, 2));
    }

public:
    RBFInterpolator(
        const vectorField& points,
        const scalarField& values,
        scalar epsilon,
        label nNeighbors,
        label polyDegree = 1)
        : points_(points),
          values_(values),
          eps_(epsilon),
          nPoints_(points.size()),
          nNeighbors_(nNeighbors),
          polyDegree_(polyDegree),
          polyDim_(getPolyDim())
    {}

    scalar interpolate(const vector& target) const
    {
        label nNeighbors = min(nPoints_, nNeighbors_);
        label totalDim = nNeighbors + polyDim_;
        
        labelList neighborIndices(nNeighbors);
        scalarField neighborDistances;
        findNearestNeighbors(target, points_, neighborIndices, neighborDistances);
        
        SquareMatrix<scalar> A(totalDim, scalar(0));
        scalarField b(totalDim, scalar(0));
        
        for (label i = 0; i < nNeighbors; i++)
        {
            label ptI = neighborIndices[i];
            b[i] = values_[ptI];
            
            for (label j = 0; j < nNeighbors; j++)
            {
                label ptJ = neighborIndices[j];
                A(i, j) = gaussianKernel(points_[ptI], points_[ptJ]);
            }
        }
        
        if (polyDim_ > 0)
        {
            scalarField polyBasis0(polyDim_);
            for (label i = 0; i < nNeighbors; i++)
            {
                label ptI = neighborIndices[i];
                polynomialBasis(points_[ptI], polyBasis0);
                for (label j = 0; j < polyDim_; j++)
                {
                    A(i, nNeighbors + j) = polyBasis0[j];
                    A(nNeighbors + j, i) = polyBasis0[j];
                }
            }
            
            for (label i = 0; i < polyDim_; i++)
            {
                A(nNeighbors + i, nNeighbors + i) = 1e-8;
            }
        }
        
        scalar regularizationParameter = 1e-9;
        for (label i = 0; i < nNeighbors; i++)
        {
            A(i, i) += regularizationParameter;
        }
        
        scalarField weights(totalDim, scalar(0));
        
        try
        {
            LUscalarMatrix solver(A);
            solver.solve(weights, b);
        }
        catch (const std::exception& e)
        {
            try
            {
                SVD svdSolver(A);
                weights = svdSolver.VSinvUt() * b;
            }
            catch (const std::exception& e2)
            {
                Info << "警告：RBF求解失败，使用最近邻值" << endl;
                return values_[neighborIndices[0]];
            }
        }
        
        scalar result = 0.0;
        for (label i = 0; i < nNeighbors; i++)
        {
            if (neighborIndices[i] >= 0)
            {
                result += weights[i] * gaussianKernel(target, points_[neighborIndices[i]]);
            }
        }
        
        if (polyDegree_ > 0)
        {
            scalarField polyBasis(polyDim_);
            polynomialBasis(target, polyBasis);
            for (label i = 0; i < polyDim_; i++)
            {
                result += weights[nNeighbors + i] * polyBasis[i];
            }
        }
        
        return result;
    }

    // 计算解析梯度
    vector gradient(const vector& target) const
    {
        label nNeighbors = min(nPoints_, nNeighbors_);
        label totalDim = nNeighbors + polyDim_;
        
        labelList neighborIndices(nNeighbors);
        scalarField neighborDistances;
        findNearestNeighbors(target, points_, neighborIndices, neighborDistances);
        
        SquareMatrix<scalar> A(totalDim, scalar(0));
        scalarField b(totalDim, scalar(0));
        
        for (label i = 0; i < nNeighbors; i++)
        {
            label ptI = neighborIndices[i];
            b[i] = values_[ptI];
            
            for (label j = 0; j < nNeighbors; j++)
            {
                label ptJ = neighborIndices[j];
                A(i, j) = gaussianKernel(points_[ptI], points_[ptJ]);
            }
        }
        
        if (polyDim_ > 0)
        {
            scalarField polyBasis0(polyDim_);
            for (label i = 0; i < nNeighbors; i++)
            {
                label ptI = neighborIndices[i];
                polynomialBasis(points_[ptI], polyBasis0);
                for (label j = 0; j < polyDim_; j++)
                {
                    A(i, nNeighbors + j) = polyBasis0[j];
                    A(nNeighbors + j, i) = polyBasis0[j];
                }
            }
            
            for (label i = 0; i < polyDim_; i++)
            {
                A(nNeighbors + i, nNeighbors + i) = 1e-8;
            }
        }
        
        scalar regularizationParameter = 1e-6;
        for (label i = 0; i < nNeighbors; i++)
        {
            A(i, i) += regularizationParameter;
        }
        
        scalarField weights(totalDim, scalar(0));
        
        try
        {
            LUscalarMatrix solver(A);
            solver.solve(weights, b);
        }
        catch (const std::exception& e)
        {
            try
            {
                SVD svdSolver(A);
                weights = svdSolver.VSinvUt() * b;
            }
            catch (const std::exception& e2)
            {
                Info << "警告：RBF求解失败，使用最近邻值" << endl;
                return vector::zero;
            }
        }
        
        scalar dfdr = 0.0;
        scalar dfdz = 0.0;
        
        for (label i = 0; i < nNeighbors; i++)
        {
            if (neighborIndices[i] >= 0)
            {
                const vector& pi = points_[neighborIndices[i]];
                scalar dx = target.x() - pi.x();
                scalar dz = target.z() - pi.z();
                scalar r2 = dx*dx + dz*dz;
                scalar expTerm = Foam::exp(-pow(eps_, 2) * r2);
                
                dfdr += weights[i] * (-2.0 * pow(eps_, 2) * dx) * expTerm;
                dfdz += weights[i] * (-2.0 * pow(eps_, 2) * dz) * expTerm;
            }
        }
        
        if (polyDegree_ > 0)
        {
            if (polyDegree_ == 1)
            {
                dfdr += weights[nNeighbors + 1];
                dfdz += weights[nNeighbors + 2];
            }
        }
        
        return vector(dfdr, 0.0, dfdz);
    }
};

// 主函数
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    // 初始化性能计时器
    clockTime timer;
    scalar startTime = timer.elapsedTime();
    
    Info << "Starting Benneke pre-processing for body force calculation" << endl;
    
    // 读取优化参数
    FluxFieldParams qrtParams, qztParams, qrzParams, qzzParams, qrrParams, qttParams;
    readFluxOptimizationParameters(
        "optimization_results/best_flux_parameters.txt",
        qrtParams, qztParams, qrzParams, qzzParams, qrrParams, qttParams
    );
    
    // 读取openFOAM_Input_Force.csv文件
    List<FluxData> fluxData;
    vectorField points;
    scalarField qrt, qzt, qrz, qzz, qrr, qtt, rhoField, lambdaField;
    readFluxData("openFOAM_Input_Force.csv", fluxData, points, qrt, qzt, qrz, qzz, qrr, qtt, rhoField, lambdaField);
    
    // 不需要从lambda场中获取lambda值，直接使用CSV文件中的值
    Info << "= w = 计算中 = w =" << endl;
    
    // 创建RBF插值器用于在数据点计算梯度（使用优化过的参数）
    RBFInterpolator rbfQrt(points, qrt, qrtParams.epsilon, qrtParams.neighbors, qrtParams.polyDegree);
    RBFInterpolator rbfQzt(points, qzt, qztParams.epsilon, qztParams.neighbors, qztParams.polyDegree);
    RBFInterpolator rbfQrz(points, qrz, qrzParams.epsilon, qrzParams.neighbors, qrzParams.polyDegree);
    RBFInterpolator rbfQzz(points, qzz, qzzParams.epsilon, qzzParams.neighbors, qzzParams.polyDegree);
    RBFInterpolator rbfQrr(points, qrr, qrrParams.epsilon, qrrParams.neighbors, qrrParams.polyDegree);
    RBFInterpolator rbfQtt(points, qtt, qttParams.epsilon, qttParams.neighbors, qttParams.polyDegree);
    
    // 创建rho和lambda的RBF插值器
    RBFInterpolator rbfRho(points, rhoField, qrtParams.epsilon, qrtParams.neighbors, qrtParams.polyDegree);
    RBFInterpolator rbfLambda(points, lambdaField, qrtParams.epsilon, qrtParams.neighbors, qrtParams.polyDegree);
    
    // 在数据点上用RBF计算梯度
    label nDataPoints = points.size();
    vectorField gradQrtData(nDataPoints);
    vectorField gradQztData(nDataPoints);
    vectorField gradQrzData(nDataPoints);
    vectorField gradQzzData(nDataPoints);
    vectorField gradQrrData(nDataPoints);
    scalarField qttData(nDataPoints);
    
    forAll(points, i)
    {
        gradQrtData[i] = rbfQrt.gradient(points[i]);
        gradQztData[i] = rbfQzt.gradient(points[i]);
        gradQrzData[i] = rbfQrz.gradient(points[i]);
        gradQzzData[i] = rbfQzz.gradient(points[i]);
        gradQrrData[i] = rbfQrr.gradient(points[i]);
        qttData[i] = rbfQtt.interpolate(points[i]);
    }
    
    // 在数据点上组装彻体力
    vectorField bodyForceData(nDataPoints);
    
    // 保存原始圆柱坐标分量用于统计
    scalarField fRValues(nDataPoints);
    scalarField fThetaValues(nDataPoints);
    scalarField fZValues(nDataPoints);
    
    Info << "在数据点上计算彻体力..." << endl;
    forAll(points, i)
    {
        scalar r = points[i].x();
        scalar rho = rhoField[i];
        scalar lambda = lambdaField[i];
        
        if (r < VSMALL) r = VSMALL;
        if (rho < VSMALL) rho = VSMALL;
        if (lambda < VSMALL) lambda = VSMALL;
        
        scalar Qtt = qttData[i];
        
        // 计算体力分量（使用CSV文件中的lambda值）- 圆柱坐标系
        scalar f_theta = (gradQrtData[i].x() + gradQztData[i].z()) / (r * r * rho * lambda);
        scalar f_z = (gradQrzData[i].x() + gradQzzData[i].z()) / (r * rho * lambda);
        scalar f_r = (gradQrrData[i].x() + gradQrzData[i].z() - Qtt) / (r * rho * lambda);
        
        // 保存圆柱坐标分量用于统计
        fRValues[i] = f_r;
        fThetaValues[i] = f_theta;
        fZValues[i] = f_z;
        
        // 数据点在子午面上（theta=0），所以笛卡尔坐标为 (f_r, f_theta, f_z)
        bodyForceData[i] = vector(f_r, f_theta, f_z);
    }
    
    // 输出原始数据点上圆柱坐标分量的统计信息
    scalar minFR = Foam::min(fRValues);
    scalar maxFR = Foam::max(fRValues);
    scalar avgFR = Foam::average(fRValues);
    
    scalar minFTheta = Foam::min(fThetaValues);
    scalar maxFTheta = Foam::max(fThetaValues);
    scalar avgFTheta = Foam::average(fThetaValues);
    
    scalar minFZ = Foam::min(fZValues);
    scalar maxFZ = Foam::max(fZValues);
    scalar avgFZ = Foam::average(fZValues);
    
    Info << "\n=== 体力统计信息 (原始数据点, 圆柱坐标系) ===" << endl;
    Info << "径向力 f_r:     min=" << minFR << ", max=" << maxFR << ", avg=" << avgFR << endl;
    Info << "周向力 f_theta: min=" << minFTheta << ", max=" << maxFTheta << ", avg=" << avgFTheta << endl;
    Info << "轴向力 f_z:     min=" << minFZ << ", max=" << maxFZ << ", avg=" << avgFZ << endl;
    Info << "====================================================" << endl;
    
    // 使用RBF插值体力场（每个分量单独插值），比IDW更精确
    // 使用与Q场相同的优化参数
    FluxFieldParams bfParams = qrtParams; // 复用已优化的参数
    RBFInterpolator rbf_fr(points, fRValues, bfParams.epsilon, bfParams.neighbors, bfParams.polyDegree);
    RBFInterpolator rbf_ftheta(points, fThetaValues, bfParams.epsilon, bfParams.neighbors, bfParams.polyDegree);
    RBFInterpolator rbf_fz(points, fZValues, bfParams.epsilon, bfParams.neighbors, bfParams.polyDegree);
    
    Info << "Created RBF interpolators for body force components" << endl;
    
    // 创建体力场
    volVectorField bodyForce
    (
        IOobject
        (
            "bodyForce",
            runTime.constant(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("bodyForce", dimAcceleration, vector(0, 0, 0))
    );
    
    // 从cellZones中获取ROTOR_FLUID区域的单元格
    const cellZone& fluidRotorZone = mesh.cellZones()["ROTOR_FLUID"];
    labelList fluidRotorCellIDs = fluidRotorZone;
    
    Info << "Found " << fluidRotorCellIDs.size() << " cells in ROTOR_FLUID cellZone" << endl;
    
    // 遍历ROTOR_FLUID中的所有单元格
    clockTime calcTimer;
    forAll(fluidRotorCellIDs, i)
    {
        label cellID = fluidRotorCellIDs[i];
        
        // 获取单元格中心点坐标
        vector cellCenter = mesh.C()[cellID];
        
        // 转换为圆柱坐标 (r, z)
        scalar r = mag(vector(cellCenter.x(), cellCenter.y(), 0.0));
        scalar z = cellCenter.z();
        
        // 构建插值目标点 (R, 0, Z)
        vector target(r, 0.0, z);
        
        // 用RBF分别插值三个分量（圆柱坐标系：r, theta, z）
        scalar fr = rbf_fr.interpolate(target);
        scalar ftheta = rbf_ftheta.interpolate(target);
        scalar fz = rbf_fz.interpolate(target);
        vector forceCyl(fr, ftheta, fz);
        
        // 将圆柱坐标系中的彻体力转换为笛卡尔坐标系
        scalar theta = Foam::atan2(cellCenter.y(), cellCenter.x());
        scalar f_x = forceCyl.x() * Foam::cos(theta) - forceCyl.y() * Foam::sin(theta);
        scalar f_y = forceCyl.x() * Foam::sin(theta) + forceCyl.y() * Foam::cos(theta);
        scalar f_z = forceCyl.z();
        
        // 设置体力场值
        bodyForce[cellID] = vector(f_x, f_y, f_z);
    }
    
    scalar calcElapsed = calcTimer.elapsedTime();
    Info << "计算时间: " << calcElapsed << " s" << endl;
    
    // 计算体力场的统计信息（只对ROTOR_FLUID区域）
    scalarField fXValues(fluidRotorCellIDs.size());
    scalarField fYValues(fluidRotorCellIDs.size());
    scalarField fZValues_cart(fluidRotorCellIDs.size());
    
    forAll(fluidRotorCellIDs, i)
    {
        label cellID = fluidRotorCellIDs[i];
        fXValues[i] = bodyForce[cellID].x();
        fYValues[i] = bodyForce[cellID].y();
        fZValues_cart[i] = bodyForce[cellID].z();
    }
    
    scalar minFX = Foam::min(fXValues);
    scalar maxFX = Foam::max(fXValues);
    scalar avgFX = Foam::average(fXValues);
    
    scalar minFY = Foam::min(fYValues);
    scalar maxFY = Foam::max(fYValues);
    scalar avgFY = Foam::average(fYValues);
    
    scalar minFZ_cart = Foam::min(fZValues_cart);
    scalar maxFZ_cart = Foam::max(fZValues_cart);
    scalar avgFZ_cart = Foam::average(fZValues_cart);
    
    Info << "\n=== 体力场统计信息 (ROTOR_FLUID区域, 笛卡尔坐标系) ===" << endl;
    Info << "X方向力 f_x: min=" << minFX << ", max=" << maxFX << ", avg=" << avgFX << endl;
    Info << "Y方向力 f_y: min=" << minFY << ", max=" << maxFY << ", avg=" << avgFY << endl;
    Info << "Z方向力 f_z: min=" << minFZ_cart << ", max=" << maxFZ_cart << ", avg=" << avgFZ_cart << endl;
    Info << "======================================================" << endl;
    
    // 写入体力场
    clockTime writeTimer;
    bodyForce.write();
    scalar writeElapsed = writeTimer.elapsedTime();
    Info << "结果写入时间: " << writeElapsed << " s" << endl;
    
    scalar totalElapsed = timer.elapsedTime() - startTime;
    Info << "总运行时间: " << totalElapsed << " s" << endl;
    
    Info << "Benneke pre-processing completed successfully!" << endl;
    
    return 0;
}

// ************************************************************************* //