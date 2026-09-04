/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \/     M anipulation  |
\*---------------------------------------------------------------------------*/

// 移动最小二乘（Moving Least Squares, MLS）彻体力前处理
// 与 Benneke_Pre_Processing.C（RBF 方法）流程一致，区别在于：
//   在每个目标点的邻域内做加权最小二乘多项式拟合（高斯权函数），
//   由局部多项式系数直接给出光滑的函数值与解析梯度，
//   再代入圆柱坐标系动量方程反演彻体力。
// MLS 不依赖 LibTorch，也不需要额外的优化超参数文件（参数内置自适应）。

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

// MLS 拟合参数
struct MLSParams
{
    label neighbors;    // 参与局部拟合的最近邻点数
    label polyDegree;   // 局部多项式阶数（1 或 2）
    scalar h;           // 高斯权函数形状参数：w(d)=exp(-(h*d/dmax)^2)
};

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
            data.lambda = values[9];
            fluxData.append(data);
        }
    }

    label nPoints = fluxData.size();
    Info << "读取到 " << nPoints << " 个数据点" << endl;

    if (nPoints == 0)
    {
        FatalErrorInFunction << "文件中没有找到有效数据点" << exit(FatalError);
    }

    points.setSize(nPoints);
    qrt.setSize(nPoints);
    qzt.setSize(nPoints);
    qrz.setSize(nPoints);
    qzz.setSize(nPoints);
    qrr.setSize(nPoints);
    qtt.setSize(nPoints);
    rhoField.setSize(nPoints);
    lambdaField.setSize(nPoints);

    forAll(fluxData, i)
    {
        const FluxData& data = fluxData[i];
        points[i] = vector(data.r, 0.0, data.z);

        // 与 RBF 版本一致的通量加权（含 r 因子与 lambda）
        qrt[i] = data.r * data.r * data.lambda * data.trt;
        qzt[i] = data.r * data.r * data.lambda * data.tzt;
        qrz[i] = data.r * data.lambda * data.trz;
        qzz[i] = data.r * data.lambda * data.tzz;
        qrr[i] = data.r * data.lambda * data.trr;
        qtt[i] = data.lambda * data.ttt;

        rhoField[i] = data.rho;
        lambdaField[i] = data.lambda;
    }

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

// 移动最小二乘插值器
// 在目标点邻域内，以目标点为原点做局部坐标加权最小二乘多项式拟合。
// 由于局部坐标原点即目标点，拟合系数 a 满足：
//   函数值 f(target) = a[0]
//   梯度     df/dx = a[1], df/dy = a[2]（高阶项在原点导数为 0）
class MLSInterpolator
{
private:
    const vectorField& points_;
    const scalarField& values_;
    label nPoints_;
    label nNeighbors_;
    label polyDegree_;
    label polyDim_;
    scalar h_;

    label getPolyDim() const
    {
        if (polyDegree_ <= 1) return 3;  // 1, x, y
        return 6;                        // 1, x, y, x^2, xy, y^2
    }

    // 局部多项式基函数（局部坐标 dx, dy）
    void polynomialBasis(scalar dx, scalar dy, scalarField& basis) const
    {
        basis.setSize(polyDim_);
        basis[0] = 1.0;
        basis[1] = dx;
        basis[2] = dy;
        if (polyDegree_ >= 2)
        {
            basis[3] = dx * dx;
            basis[4] = dx * dy;
            basis[5] = dy * dy;
        }
    }

    // 在 target 处做一次加权最小二乘拟合，返回多项式系数 a
    // （a[0]=函数值, a[1]=df/dr, a[2]=df/dz）
    bool fit(const vector& target, scalarField& a) const
    {
        label nNeighbors = min(nPoints_, nNeighbors_);
        if (nNeighbors < polyDim_)
        {
            return false; // 点数不足以确定多项式
        }

        labelList neighborIndices(nNeighbors);
        scalarField neighborDistances;
        findNearestNeighbors(target, points_, neighborIndices, neighborDistances);

        // 自适应尺度：以最远邻点距离作为权函数参考长度
        scalar dRef = Foam::max(neighborDistances[nNeighbors - 1], VSMALL);

        // 组装加权最小二乘正规方程  A a = b
        // A = P^T W P (polyDim x polyDim), b = P^T W f
        SquareMatrix<scalar> A(polyDim_, scalar(0));
        scalarField b(polyDim_, scalar(0));
        scalarField phi(polyDim_);

        for (label i = 0; i < nNeighbors; i++)
        {
            label ptI = neighborIndices[i];
            if (ptI < 0) continue;

            scalar d = neighborDistances[i];
            if (d < 1e-12)
            {
                // 目标点与数据点重合：直接返回该点值，梯度为 0
                a.setSize(polyDim_, scalar(0));
                a[0] = values_[ptI];
                return true;
            }

            // 高斯权函数
            scalar w = Foam::exp(-Foam::pow(h_ * d / dRef, 2));

            scalar dx = points_[ptI].x() - target.x();
            scalar dy = points_[ptI].z() - target.z();
            polynomialBasis(dx, dy, phi);

            scalar f = values_[ptI];

            for (label m = 0; m < polyDim_; m++)
            {
                b[m] += w * phi[m] * f;
                for (label n = 0; n < polyDim_; n++)
                {
                    A(m, n) += w * phi[m] * phi[n];
                }
            }
        }

        // 微小对角正则化，改善病态条件
        for (label m = 0; m < polyDim_; m++)
        {
            A(m, m) += 1e-10;
        }

        a.setSize(polyDim_, scalar(0));
        try
        {
            LUscalarMatrix solver(A);
            solver.solve(a, b);
        }
        catch (const std::exception&)
        {
            try
            {
                SVD svdSolver(A);
                a = svdSolver.VSinvUt() * b;
            }
            catch (const std::exception&)
            {
                return false;
            }
        }
        return true;
    }

public:
    MLSInterpolator(
        const vectorField& points,
        const scalarField& values,
        label nNeighbors = 24,
        label polyDegree = 2,
        scalar h = 2.5)
        : points_(points),
          values_(values),
          nPoints_(points.size()),
          nNeighbors_(nNeighbors),
          polyDegree_(polyDegree),
          polyDim_(getPolyDim()),
          h_(h)
    {}

    scalar interpolate(const vector& target) const
    {
        scalarField a;
        if (!fit(target, a))
        {
            labelList idx(1);
            scalarField dist;
            findNearestNeighbors(target, points_, idx, dist);
            return (idx[0] >= 0) ? values_[idx[0]] : 0.0;
        }
        return a[0]; // 局部坐标原点处函数值
    }

    // 解析梯度（圆柱子午面：x=r 方向, z=z 方向）
    vector gradient(const vector& target) const
    {
        scalarField a;
        if (!fit(target, a))
        {
            return vector::zero;
        }
        return vector(a[1], 0.0, a[2]);
    }
};

// 主函数
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    clockTime timer;
    scalar startTime = timer.elapsedTime();

    Info << "Starting MLS (Moving Least Squares) pre-processing for body force calculation" << endl;

    // MLS 参数（内置自适应，无需外部优化文件）
    MLSParams params;
    params.neighbors = 24;
    params.polyDegree = 2;
    params.h = 2.5;
    Info << "MLS 参数: neighbors=" << params.neighbors
         << ", polyDegree=" << params.polyDegree
         << ", h=" << params.h << endl;

    // 读取openFOAM_Input_Force.csv文件
    List<FluxData> fluxData;
    vectorField points;
    scalarField qrt, qzt, qrz, qzz, qrr, qtt, rhoField, lambdaField;
    readFluxData("openFOAM_Input_Force.csv", fluxData, points, qrt, qzt, qrz, qzz, qrr, qtt, rhoField, lambdaField);

    Info << "= w = 计算中 = w =" << endl;

    // 创建MLS插值器
    MLSInterpolator mlsQrt(points, qrt, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsQzt(points, qzt, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsQrz(points, qrz, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsQzz(points, qzz, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsQrr(points, qrr, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsQtt(points, qtt, params.neighbors, params.polyDegree, params.h);

    MLSInterpolator mlsRho(points, rhoField, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mlsLambda(points, lambdaField, params.neighbors, params.polyDegree, params.h);

    // 在数据点上用MLS计算梯度
    label nDataPoints = points.size();
    vectorField gradQrtData(nDataPoints);
    vectorField gradQztData(nDataPoints);
    vectorField gradQrzData(nDataPoints);
    vectorField gradQzzData(nDataPoints);
    vectorField gradQrrData(nDataPoints);
    scalarField qttData(nDataPoints);

    forAll(points, i)
    {
        gradQrtData[i] = mlsQrt.gradient(points[i]);
        gradQztData[i] = mlsQzt.gradient(points[i]);
        gradQrzData[i] = mlsQrz.gradient(points[i]);
        gradQzzData[i] = mlsQzz.gradient(points[i]);
        gradQrrData[i] = mlsQrr.gradient(points[i]);
        qttData[i] = mlsQtt.interpolate(points[i]);
    }

    // 在数据点上组装彻体力
    vectorField bodyForceData(nDataPoints);
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

        // 圆柱坐标系彻体力分量（与 RBF 版本相同的公式）
        scalar f_theta = (gradQrtData[i].x() + gradQztData[i].z()) / (r * r * rho * lambda);
        scalar f_z = (gradQrzData[i].x() + gradQzzData[i].z()) / (r * rho * lambda);
        scalar f_r = (gradQrrData[i].x() + gradQrzData[i].z() - Qtt) / (r * rho * lambda);

        fRValues[i] = f_r;
        fThetaValues[i] = f_theta;
        fZValues[i] = f_z;

        bodyForceData[i] = vector(f_r, f_theta, f_z);
    }

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

    // 用MLS插值体力场三个分量到网格
    MLSInterpolator mls_fr(points, fRValues, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mls_ftheta(points, fThetaValues, params.neighbors, params.polyDegree, params.h);
    MLSInterpolator mls_fz(points, fZValues, params.neighbors, params.polyDegree, params.h);

    Info << "Created MLS interpolators for body force components" << endl;

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

    clockTime calcTimer;
    forAll(fluidRotorCellIDs, i)
    {
        label cellID = fluidRotorCellIDs[i];

        vector cellCenter = mesh.C()[cellID];

        // 转换为圆柱坐标 (r, z)
        scalar r = mag(vector(cellCenter.x(), cellCenter.y(), 0.0));
        scalar z = cellCenter.z();

        vector target(r, 0.0, z);

        // 用MLS分别插值三个分量（圆柱坐标系：r, theta, z）
        scalar fr = mls_fr.interpolate(target);
        scalar ftheta = mls_ftheta.interpolate(target);
        scalar fz = mls_fz.interpolate(target);
        vector forceCyl(fr, ftheta, fz);

        // 圆柱坐标 -> 笛卡尔坐标
        scalar theta = Foam::atan2(cellCenter.y(), cellCenter.x());
        scalar f_x = forceCyl.x() * Foam::cos(theta) - forceCyl.y() * Foam::sin(theta);
        scalar f_y = forceCyl.x() * Foam::sin(theta) + forceCyl.y() * Foam::cos(theta);
        scalar f_z = forceCyl.z();

        bodyForce[cellID] = vector(f_x, f_y, f_z);
    }

    scalar calcElapsed = calcTimer.elapsedTime();
    Info << "计算时间: " << calcElapsed << " s" << endl;

    // 体力场统计信息（ROTOR_FLUID区域）
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

    Info << "MLS pre-processing completed successfully!" << endl;

    return 0;
}

// ************************************************************************* //
