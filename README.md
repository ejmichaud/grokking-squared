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


