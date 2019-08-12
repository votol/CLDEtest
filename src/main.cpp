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
#include "CLBLAS.h"
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

int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile(argv[1]);
    size_t num = 20000000;
    std::shared_ptr<ICLmanager> manag = std::make_shared<CLmanager>(config["properties"]);
    CLDataStorage<double> a(std::vector<double>(num, 0.7), manag);
    CLDataStorage<double> a1(std::vector<double>(num, 1.7), manag);
    CLDataStorage<double> b(10000000, manag);
    b = std::vector<double>(num, 1.3);
    CLBLAS blas(manag);

    blas.Daxpy(a, a1, 0.5);
    blas.Daxpy(a, b, -0.5);
    auto result = a.read();
    std::cout << result[100] << std::endl;
    std::string output_dir = config["properties"]
								  [CLDEtestSchema::PROPERTY_output_path].as<std::string>();
    std::vector<std::unique_ptr<IOutput> > outputs(0);
    NetCdfWriter netcdf_writer_instance(
			output_dir + "/output.nc", outputs, 0);

    return 0;
}
