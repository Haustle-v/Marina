/*
g++ -O3 -Wall -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) test_pybind.cc \
    -o test_pybind \
    -Wl,-rpath,$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    $(python3-config --ldflags --embed)

./test_pybind
*/

#include <Python.h>
#include <pybind11/embed.h>

#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

int main() {
    // 后期写到配置文件中，需要用 Py_DecodeLocale 和 PyMem_RawFree
    // python3 -c "import sys; print(sys.prefix)"
    const std::wstring py_home_base = L"/home/yefengshuo.yfs/anaconda3/envs/ray_env_python313";
    // python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))"
    const std::wstring py_stdlib_base = L"/home/yefengshuo.yfs/anaconda3/envs/ray_env_python313/lib/python3.13";
    const std::wstring py_executable = py_home_base + L"/bin/python3";
    const std::wstring py_dynload = py_stdlib_base + L"/lib-dynload";
    const std::wstring py_site = py_stdlib_base + L"/site-packages";

    const wchar_t *const py_home_w = py_home_base.c_str();
    const wchar_t *const py_executable_w = py_executable.c_str();
    const wchar_t *const py_stdlib_w = py_stdlib_base.c_str();
    const wchar_t *const py_dynload_w = py_dynload.c_str();
    const wchar_t *const py_site_w = py_site.c_str();

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.user_site_directory = 0;
    config.module_search_paths_set = 1;

    status = PyConfig_SetString(&config, &config.home, py_home_w);
    status = PyConfig_SetString(&config, &config.executable, py_executable_w);
    status = PyWideStringList_Append(&config.module_search_paths, py_stdlib_w);
    status = PyWideStringList_Append(&config.module_search_paths, py_dynload_w);
    status = PyWideStringList_Append(&config.module_search_paths, py_site_w);
    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
    
    int return_code = 0;
    {
    try {
        py::gil_scoped_acquire gil;

        std::cout << "\n=== Python / pybind Paths ===" << std::endl;
        py::module_ sys = py::module_::import("sys");
        std::cout << "sys.executable: " << py::str(sys.attr("executable")).cast<std::string>() << std::endl;
        std::cout << "sys.prefix: " << py::str(sys.attr("prefix")).cast<std::string>() << std::endl;
        std::cout << "sys.base_prefix: " << py::str(sys.attr("base_prefix")).cast<std::string>() << std::endl;
        std::cout << "sys.exec_prefix: " << py::str(sys.attr("exec_prefix")).cast<std::string>() << std::endl;
        std::cout << "sys.version: " << py::str(sys.attr("version")).cast<std::string>() << std::endl;
        std::cout << "sys.path: " << py::str(sys.attr("path")).cast<std::string>() << std::endl;

        py::module_ pybind_mod = py::module_::import("pybind11");
        std::string pybind_module_file = "N/A";
        if (py::hasattr(pybind_mod, "__file__")) {
            pybind_module_file = py::str(pybind_mod.attr("__file__")).cast<std::string>();
        }
        std::cout << "pybind11.__file__: " << pybind_module_file << std::endl;
        if (py::hasattr(pybind_mod, "get_include")) {
            std::cout << "pybind11.get_include(): " << py::str(pybind_mod.attr("get_include")()).cast<std::string>() << std::endl;
        }

        // 读取序列化的函数文件
        std::ifstream file("cloudpickle_ab_sum.pkl", std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file 'ab_sum.pkl'");
        }
        
        // 读取整个文件到内存
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        file.close();
        
        if (buffer.empty()) {
            throw std::runtime_error("File 'ab_sum.pkl' is empty");
        }
        
        // 将二进制数据转换为Python bytes对象
        py::bytes serialized_data(buffer.data(), buffer.size());
        
        // 导入cloudpickle模块并反序列化函数
        auto cloudpickle = py::module_::import("cloudpickle");
        auto func = cloudpickle.attr("loads")(serialized_data);
        
        // 示例数据1: 整数
        std::cout << "=== Test 1: Integer inputs ===" << std::endl;
        int a1 = 10, b1 = 20;
        std::cout << "func(" << a1 << ", " << b1 << ") = " << func(a1, b1).cast<double>() << std::endl;
        
        // 示例数据2: 浮点数
        std::cout << "\n=== Test 2: Float inputs ===" << std::endl;
        double a2 = 3.5, b2 = 2.7;
        std::cout << "func(" << a2 << ", " << b2 << ") = " << func(a2, b2).cast<double>() << std::endl;
        

        std::cout << "\n=== Offload to Ray ===" << std::endl;
        setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0", 1);   // 消除一个warning
        auto ray = py::module_::import("ray");
        // ray.attr("init")(py::arg("num_cpus")=4, py::arg("num_gpus")=0, py::arg("include_dashboard")=false);

        std::cout << "\n=== Ray Task Demo ===" << std::endl;
        
        // 将 func 包装为 Ray remote 函数
        auto remote_func = ray.attr("remote")(func);
        
        int a3 = 100, b3 = 200;
        auto object_ref = remote_func.attr("remote")(a3, b3);
        
        auto ray_result = ray.attr("get")(object_ref);
        std::cout << "Ray remote func(" << a3 << ", " << b3 << ") = " << ray_result.cast<double>() << std::endl;

        std::cout << "\n=== Ray Actor Demo ===" << std::endl;

        std::ifstream counter_file("cloudpickle_counter.pkl", std::ios::binary);
        if (!counter_file.is_open()) {
            throw std::runtime_error("Cannot open file 'cloudpickle_counter.pkl'");
        }
        std::vector<char> counter_buffer(std::istreambuf_iterator<char>(counter_file), {});
        counter_file.close();
        py::bytes counter_serialized_data(counter_buffer.data(), counter_buffer.size());
        
        // 反序列化 Counter 类
        auto counter_class = cloudpickle.attr("loads")(counter_serialized_data);
        
        // 将 Counter 类包装为 Ray remote Actor
        auto remote_counter_class = ray.attr("remote")(counter_class);
        
        // 创建 Actor 实例: c = Counter.remote()
        auto c = remote_counter_class.attr("remote")();
        
        // future0 = c.read.remote()
        auto future0 = c.attr("read").attr("remote")();
        
        // c.increment.remote()
        c.attr("increment").attr("remote")();
        
        // print(ray.get(future0))
        auto result0 = ray.attr("get")(future0);
        std::cout << "Initial read result: " << result0.cast<int>() << std::endl;
        
        // future1 = c.read.remote()
        auto future1 = c.attr("read").attr("remote")();
        
        // c.increment.remote()
        c.attr("increment").attr("remote")();
        
        // print(ray.get(future1))
        auto result1 = ray.attr("get")(future1);
        std::cout << "After first increment: " << result1.cast<int>() << std::endl;
        
        // future2 = c.read.remote()
        auto future2 = c.attr("read").attr("remote")();
        
        // print(ray.get(future2))
        auto result2 = ray.attr("get")(future2);
        std::cout << "After second increment: " << result2.cast<int>() << std::endl;

        std::cout << "\n=== Variable Types ===" << std::endl;
        std::cout << "remote_counter_class type: " << py::str(py::type::of(remote_counter_class)).cast<std::string>() << std::endl;
        std::cout << "c type: " << py::str(py::type::of(c)).cast<std::string>() << std::endl;
        std::cout << "future0 type: " << py::str(py::type::of(future0)).cast<std::string>() << std::endl;
        std::cout << "result0 type: " << py::str(py::type::of(result0)).cast<std::string>() << std::endl;

        ray.attr("shutdown")();
        
        std::cout << "\nSuccess! Function executed correctly." << std::endl;
        
    } catch (const py::error_already_set& e) {
        // Python异常
        std::cerr << "Python error: " << e.what() << std::endl;
        return_code = 1;
    } catch (const std::exception& e) {
        // C++异常
        std::cerr << "C++ error: " << e.what() << std::endl;
        return_code = 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return_code = 1;
    }
    }
    
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
    return return_code;
}
