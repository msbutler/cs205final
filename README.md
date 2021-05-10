## Large Scale Computer Vision for Flood Disaster Management 
## CS 205 Final Project (Spring 2021)

**Contributors: Michael Butler, Preston Ching, M. Elaine Cunha**

### Website
[https://msbutler.github.io/cs205final.github.io/](https://msbutler.github.io/cs205final.github.io/)

### Directory
- `Train/`:  folder with subdirectories for labeled and unlabeled training data
- `Figures/`:  folder containing results from performance tests
- `Presentation/`:  folder containing slides from final presentation
- `_old/`:  folder containing out-of-date versions of files
- `architecture.py`:  defines CNN architecture
- `config.py`:  configures data parameters (i.e., image resize dimensions)
- `run.py`:  trains one instance of the supervised or semisupervised model
- `semisupervised.py`:  specifies training methodology for the semisupervised model
- `supervised.py`:  specifies training methodology for the fully supervised model
- `strong_scaling.py`:  script for running strong scaling performance tests (see `Replication.md` for instructions)
- `weak_scaling.py`:  script for running weak scaling performance tests (see `Replication.md` for instructions)
- `utils.py`:  contains functions for image analysis and developing training/testing sets
