#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <thread>
#include "CLDataStorage.h"
#include "CLmanager.h"
#include "yaml-cpp/yaml.h"
#include "schema.h"
#include "DERunge4.h"
#include "PolynomialOutput.h"
#include "WienerFuncCalculator.h"
#include "NetCdfWriter.h"

using namespace clde;

void check_cl_error(cl_int err_num, const std::string& msg) {
  if(err_num != CL_SUCCESS) {
    std::cout << "[Error] OpenCL error code: " <<  err_num << " in " << msg << std::endl;
    exit(EXIT_FAILURE);
  }
}

void print_info(ICLmanager* manag)
{
    cl_device_id device = manag->device();
    char str_buffer[1024];
    cl_int err_num;
    // Get device name
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device name");
    std::cout << "CL_DEVICE_NAME: " << str_buffer << std::endl;

    // Get device hardware version
    err_num = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(str_buffer), &str_buffer, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device hardware version");
    std::cout << "CL_DEVICE_VERSION: " << str_buffer << std::endl;

    // Get device software version
    err_num = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(str_buffer), &str_buffer, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device software version");
    std::cout << "CL_DRIVER_VERSION: " << str_buffer << std::endl;

    // Get device OpenCL C version
    err_num = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(str_buffer), &str_buffer, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device OpenCL C version");
    std::cout << "CL_DEVICE_OPENCL_C_VERSION: " << str_buffer << std::endl;

    // Get device max clock frequency
    cl_uint max_clock_freq;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_freq), &max_clock_freq, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device max clock frequency");
    std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << max_clock_freq << std::endl;

    // Get device max compute units available
    cl_uint max_compute_units_available;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device max compute units available");
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << max_compute_units_available << std::endl;

    // Get device global mem size
    cl_ulong global_mem_size;
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device global mem size");
    std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: " << global_mem_size << std::endl;

    // Get device max memory available
    cl_ulong max_mem_alloc_size;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device max mem alloc size");
    std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << max_mem_alloc_size << std::endl;

    // Get device local mem size
    cl_ulong local_mem_size;
    err_num = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device local mem size");
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << local_mem_size << std::endl;

    // Get device max work group size
    size_t max_work_group_size;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device max work group size");
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << max_work_group_size << std::endl;

    // Get device max work item dim
    cl_uint max_work_item_dims;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, nullptr);
    check_cl_error(err_num, "clGetDeviceInfo: Getting device max work item dimension");
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_work_item_dims << std::endl;

}


void buildOperator(std::list<MonomialC>& in)
{
    in.push_back(MonomialC());
    in.back().coe = std::complex(-1.0,0.0);
    in.back().inInds.push_back(0);
    in.back().outInd = 0;
    in.push_back(MonomialC());
    in.back().coe = std::complex(0.0,-1.0);
    in.back().inInds.push_back(0);
    in.back().inInds.push_back(0);
    in.back().inInds.push_back(1);
    in.back().outInd = 0;
    in.push_back(MonomialC());
    in.back().coe = std::complex(1.0,0.0);
    in.back().outInd = 0;
    in.push_back(MonomialC());
    in.back().coe = std::sqrt(std::complex(0.0,1.0));
    in.back().inInds.push_back(0);
    in.back().tFunc = 0;
    in.back().outInd = 0;

    in.push_back(MonomialC());
    in.back().coe = std::complex(-1.0,0.0);
    in.back().inInds.push_back(1);
    in.back().outInd = 1;
    in.push_back(MonomialC());
    in.back().coe = std::complex(0.0,-1.0);
    in.back().inInds.push_back(1);
    in.back().inInds.push_back(1);
    in.back().inInds.push_back(0);
    in.back().outInd = 1;
    in.push_back(MonomialC());
    in.back().coe = std::complex(1.0,0.0);
    in.back().outInd = 1;
    in.push_back(MonomialC());
    in.back().coe = std::sqrt(std::complex(0.0,-1.0));
    in.back().inInds.push_back(1);
    in.back().tFunc = 1;
    in.back().outInd = 1;
}

void buildOutput(std::list<MonomialC>& in)
{
    in.push_back(MonomialC());
    in.back().coe = std::complex(1.0,0.0);
    in.back().inInds.push_back(0);
    in.back().inInds.push_back(1);
    in.back().outInd = 0;
}


void print(const std::list<Monomial>& in)
{
    for(auto it = in.begin();it != in.end();++it)
    {
        std::cout<< it->outInd<< " : " << it->coe;
        if(it->inInds .size() !=0)
            std::cout<< " : ";
        for(auto it1 = it->inInds.begin(); it1 != it->inInds.end(); ++it1)
        {
            std::cout << *it1;
            if(it1 != --it->inInds.end())
                std::cout << " , ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"----------------------"<<std::endl;
}

class outTmp: public PolynomialOutput, public IOutput
{
private:
    std::string m_name = "out";
    std::vector<size_t> dims;
public:
    template<typename _InputIterator,
         typename = std::_RequireInputIter<_InputIterator>>
    outTmp(_InputIterator begin,  _InputIterator end,
           const unsigned int& num, const OperatorDimension& dim, const unsigned int& calc_count,
           const std::shared_ptr<ICLmanager>& context): PolynomialOutput(begin,end, num, dim, calc_count, context)
    {
        dims.push_back(dim.out_dim);
    }

    virtual const std::string& GetName() override
    {
        return m_name;
    }
    virtual const std::vector<double>& GetData() override
    {
        return m_result;
    }
    virtual const std::vector<size_t>& GetDimensions() override
    {
        return dims;
    }
};

void testDerivative()
{
    Polynomial poly;
    poly.push_back(Monomial());
    poly.back().coe = 2.0;
    poly.back().inInds.push_back(0);
    poly.back().inInds.push_back(0);
    poly.back().inInds.push_back(1);
    poly.back().inInds.push_back(2);
    poly.back().inInds.push_back(3);
    poly.back().inInds.push_back(0);
    poly.back().outInd = 0;

    poly.push_back(Monomial());
    poly.back().coe = 3.0;
    poly.back().inInds.push_back(0);
    poly.back().inInds.push_back(2);
    poly.back().inInds.push_back(1);
    poly.back().inInds.push_back(0);
    poly.back().inInds.push_back(0);
    poly.back().inInds.push_back(3);
    poly.back().outInd = 0;

    std::cout<< "derivative" << std::endl;
    auto res = polynomialDerivative(poly, 5);
    for(auto it = res.begin(); it != res.end(); ++ it)
    {
        std::cout << *it <<std::endl;
    }
    std::cout << "derivative end" <<std::endl;
}

int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile(argv[1]);
    size_t out_num = 200;
    size_t traj_num = 8000;
    std::shared_ptr<ICLmanager> manag = std::make_shared<CLmanager>(config["properties"]);

    testDerivative();
    std::list<IDEOutput*> s_outputs;

    std::shared_ptr<IFuncCalculator> wiener = std::static_pointer_cast<IFuncCalculator, WienerFuncCalculator>(std::make_shared<WienerFuncCalculator>(manag));
    std::list<MonomialC> monomials;
    buildOperator(monomials);
    std::list<Monomial> real_monomials(convertMonomials(monomials));

    OperatorDimension operDim = PolynomialOperator::calculateDimension(real_monomials.begin(), real_monomials.end(), true);
    PolynomialOperator oper(real_monomials.begin(), real_monomials.end(), traj_num, operDim, manag);
    oper.setTimeFuncCalculator(wiener);

    std::list<MonomialC> monomialsOut;
    buildOutput(monomialsOut);
    std::list<Monomial> real_monomialsOut(convertMonomials(monomialsOut));
    OperatorDimension outDim = PolynomialOperator::calculateDimension(real_monomialsOut.begin(), real_monomialsOut.end(), false);
    outDim.in_dim = operDim.in_dim;
    outTmp output(real_monomialsOut.begin(), real_monomialsOut.end(), traj_num, outDim, out_num, manag);
    s_outputs.push_back(&output);

    DERunge4 calc(manag, &oper);
    calc.SetTimeStep(0.000008);
    calc.SetStepsNumber(500000);
    calc.SetOutputSteps(out_num);
    std::vector<double> init(oper.dimension().in_dim , 0.0);
    init[0] = 1.0;
    calc.SetInitState(init);
    calc.SetOutputs(s_outputs);
    //calc.calculate();

    std::string output_dir = config["properties"]
								  [CLDEtestSchema::PROPERTY_output_path].as<std::string>();
    std::vector<IOutput* > outputs(0);
    outputs.push_back(&output);
    NetCdfWriter netcdf_writer_instance(
            output_dir + "/output.nc", outputs, out_num);

    return 0;
}
