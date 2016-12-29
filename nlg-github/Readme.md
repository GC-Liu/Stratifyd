This is the nlg script for analyzing the data in a semantic topic figure, which consists of 4 parts: category, buzzword,temporal trend and geographical distribution.

The main file is: json_process_sementic_topics_alpha.py.
json_process_tempral_trend_func_version.py is a script for analyzing temporal trend data. 
json_methods.py is a script for storing some widget methods. 
All of the data are put in the data fold. 


To run the whole script, just run command line:
python3 json_process_sementic_topics_alpha.py


The input of the code is a json file containing the data and the output of the code is a paragraph of words.
As an example, the path of the json file analyzed here is: data/categories.json
The path of the corresponding figure is: data/Semantic.png
