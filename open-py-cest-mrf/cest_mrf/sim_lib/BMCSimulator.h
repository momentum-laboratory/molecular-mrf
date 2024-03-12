/* file SimPulseqSBB.h */
//#include "SimulationParameters.h"


#pragma once

#include "BMCSim.h"

#define MAX_CEST_POOLS 100

//enum CallMode { INIT, UPDATE, RUN, CLOSE, INVALID };

class BMCSimulator
{
public:

//    BMCSimulator();

    //! Constructor:
    BMCSimulator(SimulationParameters &, std::string);

//    void InitSimulation(std::string);

    Eigen::MatrixXd RunSimulation();

//    void UpdateParams(SimulationParameters);

	//! Destructor
	~BMCSimulator();


protected:
    SimulationParameters sp;
    BMCSim* simFramework = NULL;
};


