/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// OpenFOAM headers first - include all OpenFOAM stuff before LibTorch
#pragma GCC diagnostic ignored "-Wold-style-cast"
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

// Save the current namespace and then pop Foam namespace to avoid string conflict
// OpenFOAM has Foam::string which conflicts with std::string when we include libtorch
#pragma push_macro("TypeName")
#undef TypeName

// Now include LibTorch - no macro conflicts, no namespace conflicts
#include <torch/torch.h>
#include <torch/script.h>

// Restore TypeName macro
#pragma pop_macro("TypeName")

// Now we can use Foam namespace again
using namespace Foam;

// MLP inference for Benneke body force method
// - Load trained PyTorch TorchScript model
// - Direct inference on each mesh cell: NO interpolation
// - Compute body force via automatic differentiation
// - Output: bodyForce (vector field) and lambda (scalar field)
class FluxMLPInferencer
{
private:
    torch::jit::script::Module model_;

    // Normalization parameters (exported from Python training)
    scalar r_min_, r_max_, z_min_, z_max_;
    std::vector<float> y_mean_;
    std::vector<float> y_std_;

public:
    // Constructor
    FluxMLPInferencer()
    : r_min_(0), r_max_(0), z_min_(0), z_max_(0)
    {}

    // Load TorchScript model from file
    bool load(const std::string& modelPath)
    {
        try {
            model_ = torch::jit::load(modelPath);
            Info << "✓ Loaded TorchScript model: " << modelPath << endl;
            return true;
        } catch (const c10::Error& e) {
            FatalErrorInFunction
                << "Failed to load model: " << modelPath << nl
                << "Error: " << e.what() << exit(FatalError);
            return false;
        }
    }

    // Set normalization parameters (must match Python training)
    void setNormalization(
        scalar r_min, scalar r_max,
        scalar z_min, scalar z_max,
        const std::vector<float>& y_mean,
        const std::vector<float>& y_std
    )
    {
        r_min_ = r_min; r_max_ = r_max;
        z_min_ = z_min; z_max_ = z_max;
        y_mean_ = y_mean; y_std_ = y_std;
    }

    // Predict 8 flux quantities from physical coordinates (r, z)
    // METHOD B: Returns [rho, lamda_Trr, lamda_Trt, lamda_Trz, lamda_Ttt, lamda_Tzt, lamda_Tzz, lambda] in physical units
    // Network directly learns lambda*flux - no product needed at runtime
    std::vector<double> predict(scalar r_phys, scalar z_phys)
    {
        // Normalize coordinates to [0, 1] - same as Python training
        scalar r_norm = (r_phys - r_min_) / (r_max_ - r_min_);
        scalar z_norm = (z_phys - z_min_) / (z_max_ - z_min_);

        // Create input tensor
        auto x = torch::tensor(
            {static_cast<float>(r_norm), static_cast<float>(z_norm)},
            torch::kFloat32
        ).unsqueeze(0);
        
        // Force shape to [1, 2] - ensure correct shape for linear layer
        x = x.reshape({1, 2});

        // Forward pass
        auto output = model_.forward({x}).toTensor().squeeze(0);

        // Denormalize back to physical units
        std::vector<double> result(8);
        for (int i = 0; i < 8; i++) {
            result[i] = static_cast<double>(output[i].item<float>()) * y_std_[i] + y_mean_[i];
        }
        return result;
    }

    // Batch compute body force for all cells at once
    // Vectorized processing = much faster than per-cell loop
    // Still uses automatic differentiation = same precision as before
    void computeBodyForceBatch(
        const std::vector<double>& r_phys_vec,
        const std::vector<double>& z_phys_vec,
        const std::vector<double>& theta_vec,
        std::vector<double>& fx_out,
        std::vector<double>& fy_out,
        std::vector<double>& fz_out
    )
    {
        const int64_t N = r_phys_vec.size();

        // Convert all coordinates to normalized r and z, enable gradients
        torch::Tensor r_norm = torch::empty({N}, torch::kFloat32);
        torch::Tensor z_norm = torch::empty({N}, torch::kFloat32);

        for (int64_t i = 0; i < N; i++) {
            // Normalize coordinates
            float r_norm_val = static_cast<float>((r_phys_vec[i] - r_min_) / (r_max_ - r_min_));
            float z_norm_val = static_cast<float>((z_phys_vec[i] - z_min_) / (z_max_ - z_min_));
            
            // Clamp to [0, 1] to prevent extrapolation outside training data range
            // This ensures stable predictions near boundaries
            r_norm_val = std::max(0.0f, std::min(1.0f, r_norm_val));
            z_norm_val = std::max(0.0f, std::min(1.0f, z_norm_val));
            
            r_norm[i] = r_norm_val;
            z_norm[i] = z_norm_val;
        }

        r_norm.set_requires_grad(true);
        z_norm.set_requires_grad(true);

        // Stack to [N, 2] for batch forward pass
        auto x = torch::stack({r_norm, z_norm}, /*dim*/1); // [N, 2]

        // Batch forward pass - one call for all cells
        auto output = model_.forward({x}).toTensor(); // [N, 8]

        // Denormalize all outputs at once
        auto y_mean_t = torch::tensor(y_mean_, torch::kFloat32).unsqueeze(0);
        auto y_std_t = torch::tensor(y_std_, torch::kFloat32).unsqueeze(0);
        auto output_phys = output * y_std_t + y_mean_t;

        // Extract all physical quantities [N] - METHOD B: network outputs lambda*flux directly
        auto rho        = output_phys.select(1, 0);
        auto lamda_Trr  = output_phys.select(1, 1);
        auto lamda_Trt  = output_phys.select(1, 2);
        auto lamda_Trz  = output_phys.select(1, 3);
        auto lamda_Ttt  = output_phys.select(1, 4);
        auto lamda_Tzt  = output_phys.select(1, 5);
        auto lamda_Tzz  = output_phys.select(1, 6);
        auto lamda      = output_phys.select(1, 7);

        // Compute all derivatives - batch mode
        // METHOD B: network directly outputs lambda*flux - no product needed!
        // retain_graph=True for all except last

        // d(lamda_Trr)/dr (lamda_Trr is direct network output)
        lamda_Trr.sum().backward(torch::Tensor(), /*retain_graph*/true);
        auto dlamTrr_dr = r_norm.grad() / (r_max_ - r_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trz)/dz
        lamda_Trz.sum().backward(torch::Tensor(), /*retain_graph*/true);
        auto dlamTrz_dz = z_norm.grad() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trt)/dr
        lamda_Trt.sum().backward(torch::Tensor(), /*retain_graph*/true);
        auto dlamTrt_dr = r_norm.grad() / (r_max_ - r_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Tzt)/dz
        lamda_Tzt.sum().backward(torch::Tensor(), /*retain_graph*/true);
        auto dlamTzt_dz = z_norm.grad() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Tzz)/dz
        lamda_Tzz.sum().backward(torch::Tensor(), /*retain_graph*/true);
        auto dlamTzz_dz = z_norm.grad() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trz)/dr (for z-momentum)
        lamda_Trz.sum().backward();
        auto dlamTrz_dr = r_norm.grad() / (r_max_ - r_min_);

        // Benneke body force equations - vectorized
        auto denom = lamda * rho;
        denom = denom.clamp_min(1e-12);

        // r-momentum - use lamda_xxx directly (no lamda* needed)
        auto r_phys_t = torch::tensor(r_phys_vec, torch::kFloat32);
        auto term1_r = dlamTrr_dr + lamda_Trr / r_phys_t;
        auto term2_r = dlamTrz_dz;
        auto term3_r = - lamda_Ttt / r_phys_t;
        auto f_r = (term1_r + term2_r + term3_r) / denom;

        // theta-momentum
        auto term1_theta = dlamTrt_dr + lamda_Trt / r_phys_t;
        auto term2_theta = dlamTzt_dz;
        auto term3_theta = lamda_Trt / r_phys_t;
        auto f_theta = (term1_theta + term2_theta + term3_theta) / denom;

        // z-momentum
        auto term1_z = dlamTrz_dr + lamda_Trz / r_phys_t;
        auto term2_z = dlamTzz_dz;
        auto f_z = (term1_z + term2_z) / denom;

        // Convert to CPU numpy-style and apply coordinate rotation
        auto f_r_cpu = f_r.to(torch::kCPU);
        auto f_theta_cpu = f_theta.to(torch::kCPU);
        auto f_z_cpu = f_z.to(torch::kCPU);

        auto f_r_access = f_r_cpu.accessor<float, 1>();
        auto f_theta_access = f_theta_cpu.accessor<float, 1>();
        auto f_z_access = f_z_cpu.accessor<float, 1>();

        fx_out.resize(N);
        fy_out.resize(N);
        fz_out.resize(N);

        for (int64_t i = 0; i < N; i++) {
            double fr = static_cast<double>(f_r_access[i]);
            double fth = static_cast<double>(f_theta_access[i]);
            double fz_val = static_cast<double>(f_z_access[i]);
            double theta = theta_vec[i];
            fx_out[i] = fr * std::cos(theta) - fth * std::sin(theta);
            fy_out[i] = fr * std::sin(theta) + fth * std::cos(theta);
            fz_out[i] = fz_val;
        }
    }

    // Compute body force components (f_r, f_theta, f_z) using automatic differentiation
    // Same formula as Python - guarantees consistent results
    std::vector<double> computeBodyForce(scalar r_phys, scalar z_phys)
    {
        // Legacy per-cell version kept for compatibility
        // Batch version is much faster - use computeBodyForceBatch instead
        float r_norm_f = static_cast<float>((r_phys - r_min_) / (r_max_ - r_min_));
        float z_norm_f = static_cast<float>((z_phys - z_min_) / (z_max_ - z_min_));

        auto r_norm = torch::tensor({r_norm_f}, torch::kFloat32);
        auto z_norm = torch::tensor({z_norm_f}, torch::kFloat32);
        r_norm.set_requires_grad(true);
        z_norm.set_requires_grad(true);

        // Forward pass
        auto x = torch::cat({r_norm, z_norm}, 0).unsqueeze(0);
        auto output = model_.forward({x}).toTensor().squeeze(0);

        // Extract physical quantities - METHOD B: network outputs lambda*flux directly
        double rho        = output[0].item<float>() * y_std_[0] + y_mean_[0];
        double lamda_Trr  = output[1].item<float>() * y_std_[1] + y_mean_[1];
        double lamda_Trt  = output[2].item<float>() * y_std_[2] + y_mean_[2];
        double lamda_Trz  = output[3].item<float>() * y_std_[3] + y_mean_[3];
        double lamda_Ttt  = output[4].item<float>() * y_std_[4] + y_mean_[4];
        double lamda      = output[7].item<float>() * y_std_[7] + y_mean_[7];

        // Tensor for autograd - METHOD B: no product needed!
        auto lamda_Trr_T = output[1] * y_std_[1] + y_mean_[1];
        auto lamda_Trt_T = output[2] * y_std_[2] + y_mean_[2];
        auto lamda_Trz_T = output[3] * y_std_[3] + y_mean_[3];
        auto lamda_Tzt_T = output[5] * y_std_[5] + y_mean_[5];
        auto lamda_Tzz_T = output[6] * y_std_[6] + y_mean_[6];

        // Compute derivatives with automatic differentiation
        // Chain rule: d/dr_phys = d/dr_norm / (r_max - r_min)
        // METHOD B: network directly outputs lambda*flux - just differentiate directly

        // d(lamda_Trr)/dr
        // retain_graph=True because we need to do multiple backward passes on the same graph
        lamda_Trr_T.backward(torch::Tensor(), /*retain_graph*/true);
        double dlamTrr_dr = r_norm.grad()[0].item<double>() / (r_max_ - r_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trz)/dz
        lamda_Trz_T.backward(torch::Tensor(), /*retain_graph*/true);
        double dlamTrz_dz = z_norm.grad()[0].item<double>() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trt)/dr
        lamda_Trt_T.backward(torch::Tensor(), /*retain_graph*/true);
        double dlamTrt_dr = r_norm.grad()[0].item<double>() / (r_max_ - r_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Tzt)/dz
        lamda_Tzt_T.backward(torch::Tensor(), /*retain_graph*/true);
        double dlamTzt_dz = z_norm.grad()[0].item<double>() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Tzz)/dz
        lamda_Tzz_T.backward(torch::Tensor(), /*retain_graph*/true);
        double dlamTzz_dz = z_norm.grad()[0].item<double>() / (z_max_ - z_min_);
        r_norm.grad().zero_(); z_norm.grad().zero_();

        // d(lamda_Trz)/dr (for z-momentum)
        // Last backward pass - don't need to retain graph anymore
        lamda_Trz_T.backward();
        double dlamTrz_dr = r_norm.grad()[0].item<double>() / (r_max_ - r_min_);

        // =============================================================
        // Benneke body force equations - METHOD B
        // =============================================================
        // r-momentum: (1/r) d(lambda r Trr)/dr + d(lambda Trz)/dz - (lambda Ttt)/r = lambda rho f_r
        // (1/r) d(lambda r Trr)/dr = d(lambda Trr)/dr + (lambda Trr)/r
        const double term1_r = dlamTrr_dr + lamda_Trr / r_phys;
        const double term2_r = dlamTrz_dz;
        const double term3_r = - lamda_Ttt / r_phys;
        const double numerator_r = term1_r + term2_r + term3_r;
        const double denom = lamda * rho;
        const double f_r = numerator_r / denom;

        // theta-momentum: (1/r) d(lambda r Trt)/dr + d(lambda Tzt)/dz + (lambda Trt)/r = lambda rho f_theta
        const double term1_theta = dlamTrt_dr + lamda_Trt / r_phys;
        const double term2_theta = dlamTzt_dz;
        const double term3_theta = + lamda_Trt / r_phys;
        const double numerator_theta = term1_theta + term2_theta + term3_theta;
        const double f_theta = numerator_theta / denom;

        // z-momentum: (1/r) d(lambda r Trz)/dr + d(lambda Tzz)/dz = lambda rho f_z
        const double term1_z = dlamTrz_dr + lamda_Trz / r_phys;
        const double term2_z = dlamTzz_dz;
        const double numerator_z = term1_z + term2_z;
        const double f_z = numerator_z / denom;

        return {f_r, f_theta, f_z};
    }
};

// Read normalization parameters from CSV exported by Python
// Format:
//   Header line
//   r_min,r_max,z_min,z_max
//   index,name,mean,std  (8 lines for 8 channels)
void readNormalization(
    const fileName& csvPath,
    scalar& r_min, scalar& r_max,
    scalar& z_min, scalar& z_max,
    std::vector<float>& y_mean,
    std::vector<float>& y_std
)
{
    IFstream file(csvPath);
    if (!file.good()) {
        FatalErrorInFunction
            << "Cannot open normalization file: " << csvPath << exit(FatalError);
    }

    Info << "Reading normalization params from: " << csvPath << endl;

    string line;
    
    // Skip first header line
    file.getLine(line);
    
    // Read coordinate line
    file.getLine(line);
    // Remove trailing \r for Windows CRLF compatibility
    if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
        line.pop_back();
    
    std::stringstream ss(line);
    std::string token;
    
    getline(ss, token, ','); r_min = std::stod(token);
    getline(ss, token, ','); r_max = std::stod(token);
    getline(ss, token, ','); z_min = std::stod(token);
    getline(ss, token, ','); z_max = std::stod(token);
    
    // Skip second header line
    file.getLine(line);
    
    y_mean.resize(8);
    y_std.resize(8);
    
    // Read 8 data lines
    for (int i = 0; i < 8; i++) {
        file.getLine(line);
        // Remove trailing \r for Windows CRLF compatibility
        if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r')
            line.pop_back();
            
        std::stringstream ss(line);
        std::string token;
        
        int idx;
        std::string name;
        float mean, std_val;
        
        getline(ss, token, ','); idx = std::stoi(token);
        getline(ss, token, ','); name = token;
        getline(ss, token, ','); mean = std::stof(token);
        getline(ss, token, ','); std_val = std::stof(token);
        
        y_mean[idx] = mean;
        y_std[idx] = std_val;
    }

    Info << "Coordinate range: r[" << r_min << ", " << r_max << "] m, z[" << z_min << ", " << z_max << "] m" << endl;
    
    // Check that we read valid values
    if (r_max <= r_min || z_max <= z_min) {
        FatalErrorInFunction
            << "Invalid coordinate range read from CSV:" << nl
            << "  r_min = " << r_min << ", r_max = " << r_max << nl
            << "  z_min = " << z_min << ", z_max = " << z_max << nl
            << "Check CSV file content" << exit(FatalError);
    }
}

// Main executable
// Workflow:
//   1. Load trained TorchScript model
//   2. Load normalization parameters
//   3. For each cell in ROTOR_FLUID cellZone:
//      a. Get cell center (x, y, z)
//      b. Convert to cylindrical (r, z)
//      c. Direct MLP inference + autograd to get (f_r, f_theta, f_z)
//      d. Convert back to Cartesian (f_x, f_y, f_z)
//      e. Assign to bodyForce field
//   4. Write bodyForce and lambda to constant/
//
// No interpolation = No interpolation error
// Perfect for grid independence study!
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    clockTime timer;
    const scalar startTime = timer.elapsedTime();

    Info << "\n";
    Info << "============================================================\n";
    Info << "  LibTorch Direct Inference for Body Force Method\n";
    Info << "  Trained MLP -> Direct inference on mesh -> No interpolation!\n";
    Info << "============================================================\n";

    // Default file paths (generated by Python training)
    const std::string modelPath = "ANN_Output/flux_mlp_traced.pt";
    const fileName normPath = "ANN_Output/normalization_params.csv";

    // Check files exist
    if (!Foam::exists(modelPath)) {
        FatalErrorInFunction
            << "\nModel not found: " << modelPath << nl
            << "Run Python first: python ANN_TrainOneMLP.py" << nl
            << exit(FatalError);
    }
    if (!Foam::exists(normPath)) {
        FatalErrorInFunction
            << "\nNormalization file not found: " << normPath << nl
            << "Run Python first: python ANN_TrainOneMLP.py" << nl
            << exit(FatalError);
    }

    // Load normalization parameters
    scalar r_min, r_max, z_min, z_max;
    std::vector<float> y_mean(8), y_std(8);
    readNormalization(normPath, r_min, r_max, z_min, z_max, y_mean, y_std);

    // Load model and set parameters
    FluxMLPInferencer inferencer;
    inferencer.load(modelPath);
    inferencer.setNormalization(r_min, r_max, z_min, z_max, y_mean, y_std);

    // Create output fields
    volVectorField bodyForce(
        IOobject(
            "bodyForce",
            runTime.constant(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("bodyForce", dimAcceleration, vector::zero)
    );

    // Get ROTOR_FLUID cell zone (only this region needs body force)
    const cellZone& rotorZone = mesh.cellZones()["ROTOR_FLUID"];
    const labelList& cellIDs = rotorZone;
    const label nCells = cellIDs.size();

    Info << "\nFound " << nCells << " cells in ROTOR_FLUID" << endl;
    Info << "Starting direct MLP inference..." << endl;

    // Set number of threads for LibTorch
    const int nThreads = std::max(4, static_cast<int>(std::thread::hardware_concurrency()));
    torch::set_num_threads(nThreads);
    Info << "Using " << nThreads << " threads\n" << endl;

    // Batch processing: collect all coordinates first
    clockTime calcTimer;
    Info << "Collecting coordinates for " << nCells << " cells..." << endl;
    std::vector<double> r_phys_vec(nCells);
    std::vector<double> z_phys_vec(nCells);
    std::vector<double> theta_vec(nCells);

    forAll(cellIDs, i) {
        label cellID = cellIDs[i];
        const vector& cc = mesh.C()[cellID];
        const scalar r = mag(vector(cc.x(), cc.y(), 0.0));
        const scalar z = cc.z();
        const scalar theta = Foam::atan2(cc.y(), cc.x());
        r_phys_vec[i] = static_cast<double>(r);
        z_phys_vec[i] = static_cast<double>(z);
        theta_vec[i] = static_cast<double>(theta);
    }

    // Batch inference - all cells at once, vectorized by LibTorch
    Info << "Starting batch inference with automatic differentiation..." << endl;
    std::vector<double> fx_vec, fy_vec, fz_vec;
    inferencer.computeBodyForceBatch(r_phys_vec, z_phys_vec, theta_vec, fx_vec, fy_vec, fz_vec);

    // Copy results back to OpenFOAM field
    forAll(cellIDs, i) {
        label cellID = cellIDs[i];
        bodyForce[cellID] = vector(fx_vec[i], fy_vec[i], fz_vec[i]);
    }

    const scalar calcTime = calcTimer.elapsedTime();
    Info << "\nInference done in " << calcTime << " s" << endl;
    Info << "Average: " << calcTime / nCells * 1e6 << " us/cell\n" << endl;

    // Compute statistics
    scalarField fx(nCells), fy(nCells), fz(nCells);
    scalar sum_fx = 0, sum_fy = 0, sum_fz = 0;
    scalar force_x = 0, force_y = 0, force_z = 0;
    scalar totalVolume = 0;
    
    forAll(cellIDs, i) {
        label cellID = cellIDs[i];
        fx[i] = bodyForce[cellID].x();
        fy[i] = bodyForce[cellID].y();
        fz[i] = bodyForce[cellID].z();
        
        // Accumulate for average (arithmetic)
        sum_fx += fx[i];
        sum_fy += fy[i];
        sum_fz += fz[i];
        
        // Accumulate for total force (volume weighted)
        scalar V = mesh.V()[cellID];
        force_x += fx[i] * V;
        force_y += fy[i] * V;
        force_z += fz[i] * V;
        totalVolume += V;
    }

    Info << "=== Body force statistics (ROTOR_FLUID region, Cartesian coordinates) ===\n";
    Info << "Per-cell (arithmetic):\n";
    Info << "  f_x:  min=" << Foam::min(fx) << ", max=" << Foam::max(fx) << ", avg_arith=" << sum_fx/nCells << endl;
    Info << "  f_y:  min=" << Foam::min(fy) << ", max=" << Foam::max(fy) << ", avg_arith=" << sum_fy/nCells << endl;
    Info << "  f_z:  min=" << Foam::min(fz) << ", max=" << Foam::max(fz) << ", avg_arith=" << sum_fz/nCells << endl;
    Info << "\n";
    Info << "Integrated total force (volume-weighted, entire region):\n";
    Info << "  F_x = " << force_x << endl;
    Info << "  F_y = " << force_y << endl;
    Info << "  F_z = " << force_z << endl;
    Info << "  F_x / V_total = " << force_x / totalVolume << endl;
    Info << "  F_y / V_total = " << force_y / totalVolume << endl;
    Info << "=========================================================================\n" << endl;

    // Write result
    clockTime writeTimer;
    bodyForce.write();
    const scalar writeTime = writeTimer.elapsedTime();
    Info << "Wrote fields in " << writeTime << " s" << endl;

    const scalar totalTime = timer.elapsedTime() - startTime;
    Info << "Total runtime: " << totalTime << " s\n" << endl;

    Info << "Done!\n";
    Info << "  Output: constant/bodyForce (body force field)\n";

    return 0;
}

// ************************************************************************* //