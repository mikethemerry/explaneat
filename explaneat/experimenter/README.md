# Experimenter

The ExplaNEAT Experimenter is a class to help
manage ExplaNEAT experiments. It is built to keep track 
of high-level experiment configuration, data set 
management, logging, standardised results creation, etc.


## Config

[Experiment Schema](./schemas/experiment.json)

## Experiment folder structure

For each experiment, a new folder in the `results.location` folder will be created. This will have the following structure:
* `[experiment.codename]_[ymdhms]` - unique folder name per 
  * `configuration` (configuration files will be copied here)
  * `results` holds results
    * `iteration_[n]` For multiple-iteration experiments
      * `intermediate` For tracking data, extra data
      * `final` for final results
  * `logs` for holding logging files
    * `iteration_[n]` by iteration

