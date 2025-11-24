# Oil Refinery Optimization Project

This project addresses the Oil Refinery Optimization Problem, a classic Haverly Pooling Problem formulated as a Nonlinear Program (NLP). The goal is to maximize the net profit from refining crude oils while adhering to constraints related to sulfur content and product demands.

## Project Structure

The project is organized into the following directories and files:

- **model/pooling.mod**: Contains the AMPL model formulation for the optimization problem, including sets, parameters, variables, objective function, and constraints.
  
- **data/pooling.dat**: Contains the data definitions for the AMPL model, specifying values for parameters such as costs, prices, sulfur contents, demands, and variable upper bounds.

- **scripts/solve.run**: Contains the AMPL commands to load the model and data files, solve the optimization problem, and display the results.

- **neos/neos.submit.txt**: Submission file for the NEOS server, specifying the solver to be used and the model and data files for solving the optimization problem.

## Instructions for Running on NEOS Server

1. **Prepare the Files**: Ensure that the model, data, and command files are correctly set up as described above.

2. **Upload to NEOS**: Go to the NEOS server website and select the appropriate solver for nonlinear programming problems.

3. **Submit the Files**: Upload the `pooling.mod` and `pooling.dat` files along with the `neos.submit.txt` file.

4. **Review Results**: After the optimization process is complete, review the results provided by the NEOS server.

## Additional Information

For any questions or issues regarding the project, please refer to the documentation provided in the respective files or contact the project maintainer.