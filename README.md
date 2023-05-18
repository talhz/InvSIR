# InvSIR

## Run Examples

### Regression Examples

To run regression examples, first 
```bash
cd PathToParentdir/InvSIR
```
For InvSIR examples, open ```InvSIR_reg.py```, there are several functions included in the file. For example, to reproduce the result of example 1 setting 1, run 
```python3
if __name__ == "__main__":
    example1(setting=1)
```
For IRM and ERM examples, just run 
```bash
sh experiments/results_reg.sh
```
in bash.
### RMNIST Examples
The results are in ```results_img.sh```, just run 
```bash
sh experiments/results_img.sh
```
in bash to reproduce results.