#ifndef __NetCdfWriter_h_
#define __NetCdfWriter_h_
#include <string>
#include <memory>
#include <vector>
#include "OutputInterface.h"

class NetCdfWriter
{
public:
    NetCdfWriter(const std::string &, std::vector<IOutput* >&, const unsigned int&);
	~NetCdfWriter(){};
};


#endif /*__NetCdfWriter_h_*/
