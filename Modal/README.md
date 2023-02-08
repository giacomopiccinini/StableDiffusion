# SizeWise

## Installation

After having pip-installed Modal, create a new token with `modal token new`. Also, you should create a HuggingFace token. 

## Running

To run the model using Modal backend, type

```
modal run stable_diffusion.py --prompt "<PROMPT>" --samples <N_SAMPLES> --steps <N_STEPS> --batch-size <BATCH_SIZE>
```

- The `prompt` should be the description of the image we are going to generate. Make it as specific as possible. Moreover, there is lots of prompt engineering going into that. 
To get good results, it is suggested to use a few techniques, e.g. starting with a specification of the type of image "an image representing" or "a painting with". At the end of the prompt, one could also specify camera and lightning properties if necessary. 

- `samples` indicates the number of images we are generating given the same prompt. 

- `steps` indicates the number of diffusion steps to take when generating an image. 

