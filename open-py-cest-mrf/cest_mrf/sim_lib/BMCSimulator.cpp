
#include "SimulationParameters.h"
//#include "SimPulseqSBBTemplate.h"
#include "BMCSimulator.h"
//#include <matrix.h>


//#include "BlochMcConnellSolver.h"
//#include <functional>
//#include <numeric>
//#include <vector>
//#include <iostream>


//! Default Constructor
//BMCSimulator::BMCSimulator() : sp(NULL) {}

BMCSimulator::BMCSimulator(SimulationParameters &sim_params, std::string seqFileName)
{
    simFramework = new BMCSim(sim_params);

    if (!simFramework->LoadExternalSequence(seqFileName)) {
        std::cerr << "BMCSimulator::BMCSimulator", "Could not read external .seq file";
    }
}

Eigen::MatrixXd BMCSimulator::RunSimulation()
{
    simFramework->RunSimulation();

    Eigen::MatrixXd Mvec = *simFramework->GetMagnetizationVectors();

    return Mvec;

}

BMCSimulator::~BMCSimulator()
{
    if (simFramework != NULL) {
        delete simFramework;
    }
}
