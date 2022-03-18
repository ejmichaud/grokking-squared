# grokking-squared

**Overleaf**: https://www.overleaf.com/project/62105458e903f33cf985c0ef

## Tasks

### Unassigned
- [ ] Do a study of when/how individual fully-connected layers and self-attention layers can implement certain operations exactly (when things have good representations). This is to understand how believable the toy model is, which implements an operation exactly (addition). See what binary operations are in the original grokking paper and describe how they can be implemented with network layers, if you have good representations.

### Ziming
- [ ] Upload current code to this repo

### Eric

### Niklas

### Ouail

## Things to be explained

1. Why generalization happens at all and why it occurs far after the training accuracies goes up.
2. Why there is an exponential growth in the necessary training time to grok as a function of training data fraction.

Also how hyperparameters effect these things.

## Code
There is a script `/scripts/run_toy_model.py` which can be used to perform training runs on the toy model with lots 
of different configurations. It is built with [sacred](https://github.com/IDSIA/sacred). To see configuration
options, use:
```
python run_toy_model.py print_config
```
To run an experiment with a configuration other than the default, for instance changing the decoder width:
```
python run_toy_model.py -F <local results dir> run with decoder_width=30
```
The flag `-F` specifies which directory to save the experiment results to.

Runs can also be triggered programatically from another Python program, see `notebooks/execute-and-plot-script.ipynb` for an example. This is useful for performing searches over lots of different configurations (although I suppose one could write a shell script to run the script several times instead).


## Other Theories & References
- A poster from the original grokking paper, which shows that models that grok tend to have very low measures of "sharpness" in the loss landscape: `https://mathai-iclr.github.io/papers/posters/MATHAI_29_poster.png`
- Rohin Shah's explanation in the AI Alignment Newsletter:  `https://www.lesswrong.com/posts/zvWqPmQasssaAWkrj/an-159-building-agents-that-know-how-to-experiment-by#DEEP_LEARNING_`
- Quintin Pope's explanation:  https://www.lesswrong.com/posts/JFibrXBewkSDmixuo/hypothesis-gradient-descent-prefers-general-circuits
- Some responses by one of the grokking authors in this thread: `https://www.reddit.com/r/mlscaling/comments/n78584/grokking_generalization_beyond_overfitting_on/`

## Misc Resources
Some tweets on grokking:
- https://twitter.com/ericjang11/status/1480633788505812992
- https://twitter.com/PreetumNakkiran/status/1480656919496781824
- https://twitter.com/TrentonBricken/status/1480666591775887369

Reproductions:
- https://github.com/teddykoker/grokking


