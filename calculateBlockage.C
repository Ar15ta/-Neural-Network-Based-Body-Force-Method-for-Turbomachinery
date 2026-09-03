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

void readOptimizationParameters(
    const fileName& paramFileName,
    scalar& pressureEpsilon,
    label& pressureNeighbors,
    label& pressurePolyDegree,
    scalar& suctionEpsilon,
    label& suctionNeighbors,
    label& suctionPolyDegree)
{
    IFstream paramFile(paramFileName);
    if (!paramFile.good())
    {
        FatalErrorInFunction << "Cannot open file " << paramFileName << exit(FatalError);
    }
    
    // Initialize with default values first
    pressureEpsilon = 0.01;
    pressureNeighbors = 50;
    pressurePolyDegree = 1;
    suctionEpsilon = 0.01;
    suctionNeighbors = 50;
    suctionPolyDegree = 1;
    
    string line;
    bool foundPressure = false;
    bool foundSuction = false;
    
    while (paramFile.getLine(line))
    {
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();
        
        if (line.find("PRESSURE SIDE:") != string::npos)
        {
            foundPressure = true;
            foundSuction = false;
        }
        else if (line.find("SUCTION SIDE:") != string::npos)
        {
            foundSuction = true;
            foundPressure = false;
        }
        else if (line.find("Polynomial Degree:") != string::npos)
        {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos)
            {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                label degree = 0;
                ss >> degree;
                if (foundPressure) pressurePolyDegree = degree;
                else if (foundSuction) suctionPolyDegree = degree;
            }
        }
        else if (foundPressure && line.find("Epsilon:") != string::npos)
        {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos)
            {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> pressureEpsilon;
            }
        }
        else if (foundPressure && line.find("Neighbors:") != string::npos)
        {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos)
            {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> pressureNeighbors;
            }
        }
        else if (foundSuction && line.find("Epsilon:") != string::npos)
        {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos)
            {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> suctionEpsilon;
            }
        }
        else if (foundSuction && line.find("Neighbors:") != string::npos)
        {
            size_t colonPos = line.find(":");
            if (colonPos != string::npos)
            {
                string value = line.substr(colonPos + 1);
                std::stringstream ss(value);
                ss >> suctionNeighbors;
            }
        }
    }
    
    Info << "读取优化参数完成:" << endl;
    Info << "  Pressure Epsilon: " << pressureEpsilon << ", Neighbors: " << pressureNeighbors << ", PolyDegree: " << pressurePolyDegree << endl;
    Info << "  Suction Epsilon: " << suctionEpsilon << ", Neighbors: " << suctionNeighbors << ", PolyDegree: " << suctionPolyDegree << endl;
}

/**
 * @brief 存储叶片表面点的数据结构
 */
struct BladeSurfacePoint
{
    scalar r;      // 径向坐标，半径
    scalar z;      // 轴向坐标，高度
    scalar theta;  // 周向角度，度
    
    // 默认构造函数，初始化为原点
    BladeSurfacePoint() : r(0.0), z(0.0), theta(0.0) {}
    
    // 带参构造函数
    BladeSurfacePoint(scalar r, scalar z, scalar theta) 
        : r(r), z(z), theta(theta) {}
};

/**
 * @brief 查找最近的N个邻居点
 */
void findNearestNeighbors(
    const vector& target,
    const vectorField& points,
    labelList& neighborIndices,
    scalarField& neighborDistances)
{
    label nPoints = points.size();          // 数据点总数
    label nNeighbors = neighborIndices.size();  // 需要找的邻居数量
    
    // 初始化距离为极大值，索引为-1
    neighborDistances.setSize(nNeighbors, GREAT);
    neighborIndices.setSize(nNeighbors, -1);
    
    // 遍历所有数据点
    for (label i = 0; i < nPoints; i++)
    {
        // 计算当前点到目标点的欧几里得距离
        scalar dist = mag(points[i] - target);
        
        // 遍历已找到的邻居位置
        for (label j = 0; j < nNeighbors; j++)
        {
            // 如果当前距离比列表中第j个距离小
            if (dist < neighborDistances[j])
            {
                // 将第j个位置之后的元素向后移动一位
                for (label k = nNeighbors - 1; k > j; k--)
                {
                    neighborDistances[k] = neighborDistances[k-1];  // 距离向后移
                    neighborIndices[k] = neighborIndices[k-1];      // 索引向后移
                }
                // 将当前点插入到第j个位置
                neighborDistances[j] = dist;
                neighborIndices[j] = i;
                break;  // 插入完成后跳出内层循环
            }
        }
    }
}

// ========================================================================== //
// RBF插值器类定义
// ========================================================================== //

/**
 * @brief 径向基函数(RBF)插值器类
 */
class RBFInterpolator
{
private:
    const vectorField& points_;    // 原始数据点坐标 (R, 0, Z)
    const scalarField& values_;    // 对应的场值 (如theta)
    scalar eps_;                   // 高斯核形状参数，控制影响范围
    label nPoints_;                // 数据点总数
    label nNeighbors_;             // 邻居数量
    label polyDegree_;             // 多项式次数 (0=无多项式, 1=线性, 2=二次)
    
    label polyDim_;                // 多项式基函数维度
    
    /**
     * @brief 获取多项式基函数维度
     * 与Python rbf_core.py保持一致：
     * degree=0: 返回1 (常数项)
     * degree=1: 返回3 (1, x, y)
     * degree=2: 返回6 (1, x, y, x^2, xy, y^2)
     */
    label getPolyDim() const
    {
        if (polyDegree_ == 0) return 1;
        if (polyDegree_ == 1) return 3;
        if (polyDegree_ == 2) return 6;
        return 1;
    }
    
    /**
     * @brief 计算多项式基函数值
     */
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

    /**
     * @brief 计算两点之间的欧几里得距离
     */
    scalar distance(const vector& p1, const vector& p2) const
    {
        return mag(p1 - p2);
    }

    /**
     * @brief 计算高斯核函数值
     */
    scalar gaussianKernel(const vector& p, const vector& p0) const
    {
        scalar d = distance(p, p0);
        return Foam::exp(-pow(eps_ * d, 2));
    }

public:
    /**
     * @brief RBF插值器构造函数
     */
    RBFInterpolator(
        const vectorField& points,
        const scalarField& values,
        scalar epsilon,
        label nNeighbors,
        label polyDegree = 2)
        : points_(points),
          values_(values),
          eps_(epsilon),
          nPoints_(points.size()),
          nNeighbors_(nNeighbors),
          polyDegree_(polyDegree),
          polyDim_(polyDegree == 0 ? 1 : (polyDegree == 1 ? 3 : 6))
    {
        Info << "RBF interpolator initialized with epsilon = " << eps_ 
             << ", neighbors = " << nNeighbors_ 
             << ", polyDegree = " << polyDegree_ << endl;
    }

    /**
     * @brief 使用RBF插值计算目标点的场值
     */
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
};



// ========================================================================== //
// CSV文件读取函数
// ========================================================================== //
// 读取Blockage CSV文件
// ========================================================================== //

void readCSVFile(
    const fileName& inputFileName,
    List<BladeSurfacePoint>& pressureSidePoints,
    List<BladeSurfacePoint>& suctionSidePoints,
    List<BladeSurfacePoint>& allPoints)
{
    IFstream file(inputFileName);
    if (!file.good())
    {
        FatalErrorInFunction
            << "Cannot open file " << inputFileName
            << exit(FatalError);
    }

    Info << "正在读取文件: " << inputFileName << endl;

    string line;
    label lineNumber = 0;
    
    // 跳过前6行说明
    while (file.getLine(line) && lineNumber < 6)
    {
        lineNumber++;
    }
    
    // 读取第一组数据
    List<BladeSurfacePoint> group1Points;
    
    while (file.getLine(line))
    {
        lineNumber++;
        if (line.empty()) break;
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();

        if (line.find(',') == string::npos || line.find("[") != string::npos)
            break;

        std::stringstream ss(line);
        std::string token;
        scalar r = 0, theta = 0, z = 0;
        label col = 0;

        while (std::getline(ss, token, ',') && col < 3)
        {
            if (!token.empty())
            {
                try { scalar val = std::stod(token); if (col == 0) r = val; else if (col == 1) theta = val; else if (col == 2) z = val; } catch (...) { continue; }
            }
            col++;
        }

        if (r > 0 && r < 10 && z > -1000 && z < 1000 && theta > -360 && theta < 360)
        {
            group1Points.append(BladeSurfacePoint(r, z, theta));
        }
    }

    label group1Count = group1Points.size();
    Info << "第一组数据读取完成: " << group1Count << " 点" << endl;

    List<BladeSurfacePoint> group2Points;
    
    Info << "开始读取第二组数据..." << endl;
    while (file.getLine(line))
    {
        lineNumber++;
        if (line.empty()) continue;
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();

        if (line.find("[Data]") != string::npos || line.find("Radius") != string::npos)
            continue;

        std::stringstream ss(line);
        std::string token;
        scalar r = 0, theta = 0, z = 0;
        label col = 0;

        while (std::getline(ss, token, ',') && col < 3)
        {
            if (!token.empty())
            {
                try { scalar val = std::stod(token); if (col == 0) r = val; else if (col == 1) theta = val; else if (col == 2) z = val; } catch (...) { continue; }
            }
            col++;
        }

        if (r > 0 && r < 10 && z > -1000 && z < 1000 && theta > -360 && theta < 360)
        {
            group2Points.append(BladeSurfacePoint(r, z, theta));
        }
    }

    label group2Count = group2Points.size();
    Info << "第二组数据读取完成: " << group2Count << " 点" << endl;

    label totalPoints = group1Count + group2Count;
    Info << "总共读取到 " << totalPoints << " 个数据点" << endl;
    
    if (totalPoints == 0)
    {
        FatalErrorInFunction << "文件中没有找到有效数据点" << exit(FatalError);
    }

    if (group1Count <= group2Count)
    {
        pressureSidePoints = group1Points;
        suctionSidePoints = group2Points;
    }
    else
    {
        pressureSidePoints = group2Points;
        suctionSidePoints = group1Points;
    }
    
    if (pressureSidePoints.size() == 0 || suctionSidePoints.size() == 0)
    {
        FatalErrorInFunction << "数据分割失败，压力面或吸力面数据为空" << exit(FatalError);
    }
    
    allPoints.append(group1Points);
    allPoints.append(group2Points);
}

// 辅助函数：对边界点按半径排序
void sortBoundaryPointsByRadius(List<vector>& points)
{
    Foam::sort(points, [](const vector& a, const vector& b) {
        return a.x() < b.x();
    });
}

// 辅助函数：使用二分查找根据半径r查找最接近的边界点的Z值
scalar findClosestZWithBinarySearch(const List<vector>& sortedPoints, scalar r)
{
    if (sortedPoints.empty()) return 0.0;
    
    label low = 0;
    label high = sortedPoints.size() - 1;
    label closestIndex = 0;
    scalar minDist = GREAT;
    
    while (low <= high)
    {
        label mid = (low + high) / 2;
        scalar rMid = sortedPoints[mid].x();
        scalar dist = mag(r - rMid);
        
        if (dist < minDist)
        {
            minDist = dist;
            closestIndex = mid;
        }
        
        if (rMid < r)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    
    return sortedPoints[closestIndex].z();
}

// ========================================================================== //
// 读取和写出Bennake CSV文件，添加lambda列
// ========================================================================== //

struct BennakeData
{
    scalar R, Rho, Trr, Trt, Trz, Ttt, Tzt, Tzz, Z, Lamda;
    bool inBlade;
};

void readAndProcessBennakeCSV(
    const fileName& inputFileName,
    const fileName& outputFileName,
    const RBFInterpolator& pressureRBF,
    const RBFInterpolator& suctionRBF,
    const scalar N,
    const scalar degreeToRad,
    const scalar twoPi,
    const List<vector>& leadingEdgePoints,
    const List<vector>& trailingEdgePoints)
{
    IFstream bennakeFile(inputFileName);
    if (!bennakeFile.good())
    {
        FatalErrorInFunction << "Cannot open file " << inputFileName << exit(FatalError);
    }

    Info << "读取 " << inputFileName << " ..." << endl;

    List<BennakeData> dataList;
    string line;
    label lineNumber = 0;

    while (bennakeFile.getLine(line))
    {
        lineNumber++;
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();

        // 跳过空行
        if (line.empty()) continue;

        // 跳过包含方括号的说明行（如 [Name], [Data]）
        if (line.find('[') != string::npos) continue;

        // 如果行包含 "Radius" 或 "Rho"，这是表头行，跳过
        if (line.find("Radius") != string::npos || line.find("Rho") != string::npos)
            continue;

        std::stringstream ss(line);
        std::string token;
        scalar values[9];
        label col = 0;

        while (std::getline(ss, token, ',') && col < 9)
        {
            try { values[col] = std::stod(token); } catch (...) { values[col] = 0; }
            col++;
        }

        if (col >= 9)
        {
            if (values[0] < 1e-10) continue;
            
            BennakeData data;
            data.R = values[0];
            data.Rho = values[1];
            data.Trr = values[2];
            data.Trt = values[3];
            data.Trz = values[4];
            data.Ttt = values[5];
            data.Tzt = values[6];
            data.Tzz = values[7];
            data.Z = values[8];
            data.Lamda = 1.0;
            data.inBlade = false;
            dataList.append(data);
        }
    }

    Info << "读取到 " << dataList.size() << " 行数据" << endl;

    forAll(dataList, i)
    {
        scalar r = dataList[i].R;
        scalar z = dataList[i].Z;
        
        if (!leadingEdgePoints.empty() && !trailingEdgePoints.empty())
        {
            scalar zLE = findClosestZWithBinarySearch(leadingEdgePoints, r);
            scalar zTE = findClosestZWithBinarySearch(trailingEdgePoints, r);
            scalar zMinBlade = Foam::min(zLE, zTE);
            scalar zMaxBlade = Foam::max(zLE, zTE);
            dataList[i].inBlade = (z >= zMinBlade) && (z <= zMaxBlade);
        }
        else
        {
            dataList[i].inBlade = false;
        }
    }

    forAll(dataList, i)
    {
        if (!dataList[i].inBlade)
        {
            dataList[i].Lamda = 1.0;
            continue;
        }
        
        scalar r = dataList[i].R;
        scalar z = dataList[i].Z;
        vector target(r, 0.0, z);
        scalar thetaPressure = pressureRBF.interpolate(target);
        scalar thetaSuction = suctionRBF.interpolate(target);
        scalar deltaTheta = Foam::mag(thetaPressure - thetaSuction);
        // 处理周期性：theta范围是-180~180或0~360，如果差值超过180度
        // 说明跨过了±180或0~360边界，应该取小的那个差
        if (deltaTheta > 180.0)
        {
            deltaTheta = 360.0 - deltaTheta;
        }
        scalar deltaThetaRad = deltaTheta * degreeToRad;
        scalar b = (deltaThetaRad * N) / twoPi;
        scalar alpha = Foam::max(0.0, Foam::min(1.0, b));
        dataList[i].Lamda = 1.0 - alpha;
    }

    scalar validCount = 0;
    scalar minLambda = GREAT;
    scalar maxLambda = -GREAT;
    scalar sumLambda = 0;
    forAll(dataList, i)
    {
        if (dataList[i].inBlade)
        {
            validCount++;
            minLambda = Foam::min(minLambda, dataList[i].Lamda);
            maxLambda = Foam::max(maxLambda, dataList[i].Lamda);
            sumLambda += dataList[i].Lamda;
        }
    }
    scalar avgLambda = validCount > 0 ? sumLambda / validCount : 0;

    Info << "\n=== 将lambda场写入通量文件...\n统计信息 (有效点数: " << validCount << ") ===" << endl;
    Info << "最小值: " << minLambda << endl;
    Info << "最大值: " << maxLambda << endl;
    Info << "平均值: " << avgLambda << endl;
    Info << "=====================" << endl;

    OFstream outputFile(outputFileName);
    if (!outputFile.good())
    {
        FatalErrorInFunction << "Cannot create file " << outputFileName << exit(FatalError);
    }

    outputFile.precision(15);
    outputFile << "R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda" << endl;

    forAll(dataList, i)
    {
        outputFile << dataList[i].R << ","
                   << dataList[i].Rho << ","
                   << dataList[i].Trr << ","
                   << dataList[i].Trt << ","
                   << dataList[i].Trz << ","
                   << dataList[i].Ttt << ","
                   << dataList[i].Tzt << ","
                   << dataList[i].Tzz << ","
                   << dataList[i].Z << ","
                   << dataList[i].Lamda << endl;
    }

    Info << "输出文件: " << outputFileName << endl;
}

// ========================================================================== //
// 主函数
// ========================================================================== //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    // 初始化性能计时器
    clockTime timer;
    scalar startTime = timer.elapsedTime();
    
    Info << "Starting blockage factor calculation for ROTOR_FLUID cells" << endl;
    
    // 输入输出文件名
    fileName inputFileName = "CFX_Output_Blockage.csv";
    
    // 声明数据存储
    List<BladeSurfacePoint> pressureSidePoints;  // 压力面数据
    List<BladeSurfacePoint> suctionSidePoints;   // 吸力面数据
    List<BladeSurfacePoint> allPoints;           // 所有数据点（按顺序存储）
    
    // 读取CSV文件
    clockTime readTimer;
    readCSVFile(inputFileName, pressureSidePoints, suctionSidePoints, allPoints);
    scalar readElapsed = readTimer.elapsedTime();
    Info << "文件读取时间: " << readElapsed << " s" << endl;
    
    // 构建压力面的RBF插值器
    label nPressure = pressureSidePoints.size();
    vectorField pressurePoints(nPressure);
    scalarField pressureTheta(nPressure);
    
    forAll(pressureSidePoints, i)
    {
        pressurePoints[i] = vector(pressureSidePoints[i].r, 0.0, pressureSidePoints[i].z);
        pressureTheta[i] = pressureSidePoints[i].theta;
    }
    
    // 统计叶片表面数据点的R,Z范围
    scalar bladeRMin = GREAT, bladeRMax = -GREAT;
    scalar bladeZMin = GREAT, bladeZMax = -GREAT;
    forAll(pressureSidePoints, i)
    {
        bladeRMin = Foam::min(bladeRMin, pressureSidePoints[i].r);
        bladeRMax = Foam::max(bladeRMax, pressureSidePoints[i].r);
        bladeZMin = Foam::min(bladeZMin, pressureSidePoints[i].z);
        bladeZMax = Foam::max(bladeZMax, pressureSidePoints[i].z);
    }
    forAll(suctionSidePoints, i)
    {
        bladeRMin = Foam::min(bladeRMin, suctionSidePoints[i].r);
        bladeRMax = Foam::max(bladeRMax, suctionSidePoints[i].r);
        bladeZMin = Foam::min(bladeZMin, suctionSidePoints[i].z);
        bladeZMax = Foam::max(bladeZMax, suctionSidePoints[i].z);
    }
    Info << "叶片表面数据 R范围: [" << bladeRMin << ", " << bladeRMax << "]" << endl;
    Info << "叶片表面数据 Z范围: [" << bladeZMin << ", " << bladeZMax << "]" << endl;
    
    // 使用优化结果中的参数
    scalar pressureEpsilon = 0, suctionEpsilon = 0;
    label pressureNeighbors = 0, suctionNeighbors = 0;
    label pressurePolyDegree = 2, suctionPolyDegree = 2;
    
    readOptimizationParameters(
        "optimization_results/best_parameters_separate.txt",
        pressureEpsilon,
        pressureNeighbors,
        pressurePolyDegree,
        suctionEpsilon,
        suctionNeighbors,
        suctionPolyDegree
    );
    
    RBFInterpolator pressureRBF(pressurePoints, pressureTheta, pressureEpsilon, pressureNeighbors, pressurePolyDegree);
    
    // 构建吸力面的RBF插值器
    label nSuction = suctionSidePoints.size();
    vectorField suctionPoints(nSuction);
    scalarField suctionTheta(nSuction);
    
    forAll(suctionSidePoints, i)
    {
        suctionPoints[i] = vector(suctionSidePoints[i].r, 0.0, suctionSidePoints[i].z);
        suctionTheta[i] = suctionSidePoints[i].theta;
    }
    
    RBFInterpolator suctionRBF(suctionPoints, suctionTheta, suctionEpsilon, suctionNeighbors, suctionPolyDegree);
    
    // 堵塞因子计算参数
    const scalar N = 36.0;                                     // 叶片数
    const scalar twoPi = 2.0 * Foam::constant::mathematical::pi;  // 2π
    const scalar degreeToRad = Foam::constant::mathematical::pi / 180.0;  // 度转弧度
    
    // 直接计算lambda场，不再计算alpha.volume
    volScalarField lambda
    (
        IOobject
        (
            "lambda",
            runTime.constant(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("lambda", dimless, 1.0)
    );
    
    // 从cellZones中获取FLUID_ROTOR区域的单元格
    const cellZone& fluidRotorZone = mesh.cellZones()["ROTOR_FLUID"];
    labelList fluidRotorCellIDs = fluidRotorZone; // 直接获取单元格ID列表
    
    Info << "Found " << fluidRotorCellIDs.size() << " cells in ROTOR_FLUID cellZone" << endl;
    
    // 获取ROTOR_TO_IN和ROTOR_TO_OUT边界的点
    List<vector> leadingEdgePoints;  // 前缘点 (ROTOR_TO_IN)
    List<vector> trailingEdgePoints; // 尾缘点 (ROTOR_TO_OUT)
    
    const polyBoundaryMesh& boundaries = mesh.boundaryMesh();
    
    // 处理ROTOR_TO_IN边界（前缘）
    label inPatchID = -1;
    forAll(boundaries, patchI)
    {
        if (boundaries[patchI].name() == "ROTOR_TO_IN")
        {
            inPatchID = patchI;
            break;
        }
    }
    if (inPatchID != -1)
    {
        const polyPatch& inPatch = boundaries[inPatchID];
        forAll(inPatch, faceI)
        {
            label faceID = inPatch.start() + faceI;
            vector faceCenter = mesh.faceCentres()[faceID];
            scalar r = mag(vector(faceCenter.x(), faceCenter.y(), 0.0));
            scalar z = faceCenter.z();
            leadingEdgePoints.append(vector(r, 0.0, z));
        }
        Info << "Found " << leadingEdgePoints.size() << " points on ROTOR_TO_IN boundary" << endl;
    }
    else
    {
        Info << "Warning: ROTOR_TO_IN boundary not found" << endl;
    }
    
    // 处理ROTOR_TO_OUT边界（尾缘）
    label outPatchID = -1;
    forAll(boundaries, patchI)
    {
        if (boundaries[patchI].name() == "ROTOR_TO_OUT")
        {
            outPatchID = patchI;
            break;
        }
    }
    if (outPatchID != -1)
    {
        const polyPatch& outPatch = boundaries[outPatchID];
        forAll(outPatch, faceI)
        {
            label faceID = outPatch.start() + faceI;
            vector faceCenter = mesh.faceCentres()[faceID];
            scalar r = mag(vector(faceCenter.x(), faceCenter.y(), 0.0));
            scalar z = faceCenter.z();
            trailingEdgePoints.append(vector(r, 0.0, z));
        }
        Info << "Found " << trailingEdgePoints.size() << " points on ROTOR_TO_OUT boundary" << endl;
    }
    else
    {
        Info << "Warning: ROTOR_TO_OUT boundary not found" << endl;
    }
    
    // 对边界点按半径排序，以便后续使用二分查找
    sortBoundaryPointsByRadius(leadingEdgePoints);
    sortBoundaryPointsByRadius(trailingEdgePoints);
    Info << "Boundary points sorted by radius" << endl;
    
    scalar rMin = GREAT, rMax = -GREAT;
    scalar zMin = GREAT, zMax = -GREAT;
    
    // 遍历FLUID_ROTOR中的所有单元格
    clockTime calcTimer;
    
    label countInBlade = 0;
    scalar sumLambda = 0.0;
    scalar minLambdaInBlade = GREAT;
    scalar maxLambdaInBlade = -GREAT;

    forAll(fluidRotorCellIDs, i)
    {
        label cellID = fluidRotorCellIDs[i];
        
        // 获取单元格中心点坐标
        vector cellCenter = mesh.C()[cellID];
        
        // 转换为圆柱坐标 (r, z)
        scalar r = mag(vector(cellCenter.x(), cellCenter.y(), 0.0));
        scalar z = cellCenter.z();
        
        rMin = Foam::min(rMin, r);
        rMax = Foam::max(rMax, r);
        zMin = Foam::min(zMin, z);
        zMax = Foam::max(zMax, z);
        
        // 检查点是否在叶片范围内
        bool inBlade = false;
        if (!leadingEdgePoints.empty() && !trailingEdgePoints.empty())
        {
            // 使用二分查找寻找最接近的前缘和尾缘点的Z值
            scalar zLE = findClosestZWithBinarySearch(leadingEdgePoints, r);
            scalar zTE = findClosestZWithBinarySearch(trailingEdgePoints, r);
            
            // 检查z是否在前缘和尾缘的Z范围内
            scalar zMinBlade = Foam::min(zLE, zTE);
            scalar zMaxBlade = Foam::max(zLE, zTE);
            inBlade = (z >= zMinBlade) && (z <= zMaxBlade);
        }
        
        if (inBlade)
        {
            // 构建插值目标点 (R, 0, Z)
            vector target(r, 0.0, z);
            
            // RBF插值得到theta_pressure和theta_suction
            scalar thetaPressure = pressureRBF.interpolate(target);
            scalar thetaSuction = suctionRBF.interpolate(target);
            
            // 计算角度差（度）
            scalar deltaTheta = Foam::mag(thetaPressure - thetaSuction);
            // 处理周期性：theta范围是-180~180或0~360，如果差值超过180度
            // 说明跨过了±180或0~360边界，应该取小的那个差
            if (deltaTheta > 180.0)
            {
                deltaTheta = 360.0 - deltaTheta;
            }
            
            // 转换为弧度并计算堵塞因子
            scalar deltaThetaRad = deltaTheta * degreeToRad;
            scalar b = (deltaThetaRad * N) / twoPi;
            
            // 限制在[0, 1]范围内
            scalar alpha = Foam::max(0.0, Foam::min(1.0, b));
            
            // 直接计算lambda = 1 - alpha
            scalar lambdaValue = 1.0 - alpha;
            lambda[cellID] = lambdaValue;

            // 统计信息更新
            countInBlade++;
            sumLambda += lambdaValue;
            minLambdaInBlade = Foam::min(minLambdaInBlade, lambdaValue);
            maxLambdaInBlade = Foam::max(maxLambdaInBlade, lambdaValue);
        }
        else
        {
            // 叶片范围外，设置为1.0
            lambda[cellID] = 1.0;
        }
    }
    
    Info << "ROTOR_FLUID R范围: [" << rMin << ", " << rMax << "]" << endl;
    Info << "ROTOR_FLUID Z范围: [" << zMin << ", " << zMax << "]" << endl;
    
    scalar calcElapsed = calcTimer.elapsedTime();
    Info << "计算时间: " << calcElapsed << " s" << endl;
    
    // 计算lambda场的统计信息
    scalar avgLambdaInBlade = countInBlade > 0 ? sumLambda / countInBlade : 0;
    
    Info << "\n=== lambda场统计信息 (叶片内部单元数: " << countInBlade << ") ===" << endl;
    Info << "最小值: " << (countInBlade > 0 ? minLambdaInBlade : 1.0) << endl;
    Info << "最大值: " << (countInBlade > 0 ? maxLambdaInBlade : 1.0) << endl;
    Info << "平均值: " << avgLambdaInBlade << endl;
    Info << "==========================================================" << endl;
    Info << "ROTOR_FLUID cell数: " << fluidRotorCellIDs.size() << endl;
    
    // ========================================================================
    // 对指定边界进行插值
    // ========================================================================
    List<word> boundaryNames;
    boundaryNames.append("ROTOR_HUB");
    boundaryNames.append("ROTOR_TO_IN");
    boundaryNames.append("ROTOR_TO_OUT");
    
    label totalBoundaryFaces = 0;
    label boundaryFacesInBlade = 0;
    
    Info << "\n开始处理边界场插值..." << endl;
    
    forAll(boundaryNames, bNameI)
    {
        const word& patchName = boundaryNames[bNameI];
        label patchID = -1;
        
        forAll(boundaries, patchI)
        {
            if (boundaries[patchI].name() == patchName)
            {
                patchID = patchI;
                break;
            }
        }
        
        if (patchID != -1)
        {
            const polyPatch& patch = boundaries[patchID];
            label nFaces = patch.size();
            totalBoundaryFaces += nFaces;
            
            Info << "  处理边界 " << patchName << " (" << nFaces << " 个面)" << endl;
            
            forAll(patch, faceI)
            {
                // 获取面中心坐标
                vector faceCenter = patch.faceCentres()[faceI];
                
                // 转换为圆柱坐标 (r, z)
                scalar r = mag(vector(faceCenter.x(), faceCenter.y(), 0.0));
                scalar z = faceCenter.z();
                
                // 检查点是否在叶片范围内
                bool inBlade = false;
                if (!leadingEdgePoints.empty() && !trailingEdgePoints.empty())
                {
                    scalar zLE = findClosestZWithBinarySearch(leadingEdgePoints, r);
                    scalar zTE = findClosestZWithBinarySearch(trailingEdgePoints, r);
                    scalar zMinBlade = Foam::min(zLE, zTE);
                    scalar zMaxBlade = Foam::max(zLE, zTE);
                    inBlade = (z >= zMinBlade) && (z <= zMaxBlade);
                }
                
                scalar lambdaValue = 1.0;
                
                if (inBlade)
                {
                    // 构建插值目标点 (R, 0, Z)
                    vector target(r, 0.0, z);
                    
                    // RBF插值得到theta_pressure和theta_suction
                    scalar thetaPressure = pressureRBF.interpolate(target);
                    scalar thetaSuction = suctionRBF.interpolate(target);
                    
                    // 计算角度差（度）
                    scalar deltaTheta = Foam::mag(thetaPressure - thetaSuction);
                    // 处理周期性：如果差值超过180度，取小的那个差
                    if (deltaTheta > 180.0)
                    {
                        deltaTheta = 360.0 - deltaTheta;
                    }
                    
                    // 转换为弧度并计算堵塞因子
                    scalar deltaThetaRad = deltaTheta * degreeToRad;
                    scalar b = (deltaThetaRad * N) / twoPi;
                    
                    // 限制在[0, 1]范围内
                    scalar alpha = Foam::max(0.0, Foam::min(1.0, b));
                    
                    // 直接计算lambda = 1 - alpha
                    lambdaValue = 1.0 - alpha;
                    boundaryFacesInBlade++;
                }
                
                // 设置边界patch上的值
                lambda.boundaryFieldRef()[patchID][faceI] = lambdaValue;
            }
        }
        else
        {
            Info << "  警告: 边界 " << patchName << " 未找到" << endl;
        }
    }
    
    Info << "边界插值完成: 总共 " << totalBoundaryFaces << " 个面，其中 " 
         << boundaryFacesInBlade << " 个面在叶片范围内" << endl;
    
    // 只保留lambda场，移除梯度场计算
    
    // 只写入lambda场
    clockTime writeTimer;
    lambda.write();
    scalar writeElapsed = writeTimer.elapsedTime();
    Info << "结果写入时间: " << writeElapsed << " s" << endl;
    
    // 处理Bennake通量数据，添加lambda列并输出openFOAM_Input_Force.csv
    clockTime csvTimer;
    readAndProcessBennakeCSV(
        "CFX_Output_Benneke_Flux.csv",
        "openFOAM_Input_Force.csv",
        pressureRBF,
        suctionRBF,
        N,
        degreeToRad,
        twoPi,
        leadingEdgePoints,
        trailingEdgePoints
    );
    scalar csvElapsed = csvTimer.elapsedTime();
    Info << "CSV处理时间: " << csvElapsed << " s" << endl;
    
    scalar totalElapsed = timer.elapsedTime() - startTime;
    Info << "总运行时间: " << totalElapsed << " s" << endl;
    
    Info << "Blockage factor calculation completed successfully!" << endl;
    
    return 0;
}

// ************************************************************************* //