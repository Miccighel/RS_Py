#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <boost/program_options.hpp>

#include "utils.hpp"

int main(int argc, const char* argv[])
{
#if !defined(NDEBUG)
    std::cout << "This code is running in DEBUG mode" << std::endl;
#endif

    namespace po = boost::program_options;
    po::options_description description("Usage:");
    description.add_options()
        ("help,h", "Display this help message")
        ("ub", po::value<int>(), "Upper bound (excl)")
        ("beta", po::value<float>()->default_value(1.5), "Beta")
        ("size,s", po::value<int>(), "Number of samples");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        std::cout << description;
    }
    if (!vm.count("ub") || !vm.count("size"))
    {
        std::cout << "Error: Missing upper or number of samples" << std::endl;
        return 1;
    }

#if !defined(NDEBUG)
    std::cout << "Upper bound " << vm["ub"].as<int>() << std::endl;
    std::cout << "Beta " << vm["beta"].as<float>() << std::endl;
    std::cout << "Samples " << vm["size"].as<int>() << std::endl;
#endif

    int lb = 1;
    int ub = vm["ub"].as<int>();
    int span_size = ub - lb + 1;
    float beta = vm["beta"].as<float>();
    int n = vm["size"].as<int>();

    std::vector<double> prefix_sum;
    std::vector<double> distribution, power_law;
    std::vector<int> samples;

    distribution.resize(span_size);
    prefix_sum.resize(distribution.size());
    for(size_t i = 0; i < distribution.size(); ++i)
    {
        distribution[i] = pow(i+1, -beta);
        prefix_sum[i] = distribution[i];
        if(i != 0)  
        {
            prefix_sum[i] += prefix_sum[i-1];
        }
    }
    power_law.resize(span_size);
    for(size_t i = 0; i < power_law.size(); ++i)
    {
        power_law[i] = prefix_sum[i] / prefix_sum.back();
    }

    std::cout << "[";
    for(size_t i = 0; i < n; ++i)
    {
        int x = sample(power_law);
        std::cout << x << ",";
    }
    std::cout << "]" << std::endl;


#if !defined(NDEBUG)
    std::cout << "This code is running in DEBUG mode" << std::endl;
#endif
    return 0;
}