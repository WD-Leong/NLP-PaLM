# NLP-PaLM

This repository trains a Pathways Language Model (PaLM) based on the paper [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311). To train the model using the [movie dialog dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), first run the script
```
python process_movie_dialogs_subword.py
```
to obtain the sub-word vocabulary. Then run
```
python train_movie_dialog_sw_tf_ver2_palm.py
```
or 
```
python train_movie_dialog_sw_torch_palm.py
```
to train the model in Tensorflow-2 or PyTorch respectively. After the model is trained, the script
```
python infer_movie_dialog_sw_tf_ver2_palm.py
```
or
```
python infer_movie_dialog_sw_torch_palm.py
```
can be run to perform inference of the trained model.

