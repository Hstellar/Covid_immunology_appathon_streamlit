# COVID19-Immunology-App-a-thon
This repository contains results of competition by PrecisionFDA on 'COVID-19 Precision Immunology App-a-thon'<br>
Work was done in team DS_CliqueprecisionFDA_team_submission_1. Team members: Priya Bharathi Kandanala, Miracle Rindani, Hena Ghonia<br>

#### Details for installation required:
Unzip precisionFDA folder<br>
Install anaconda : https://docs.anaconda.com/anaconda/install/<br>
create new environment with package.txt by typing in anaconda command prompt:<br> 
`conda create --name preFDA --file package.txt`<br>
`activate preFDA`<br>
Type streamlit run command in anaconda prompt to see app in local environment: `streamlit run appathon.py`<br>
In browser a window will open with displayed results.<br>


##### To understand how ETL process was done(optional)
install jupyter lab by typing follwing in anaconda command prompt:<br>
`conda install -c conda-forge jupyterlab`<br>
`jupyter-lab`
In data folder place Adaptive&ISB_Metadata.csv and adaptive-ISB-combined.tsv(unzip this file.)<br>
Open file 'fork-of-precisionfda-immunology-appathon.ipynb' in jupyter lab and run each cell.<br>
Now processed data is in data folder with naming convention v_call.parquet, j_call.parquet, d_call.parquet.<br>
Open another anaconda command prompt window and change directory to `cd COVID19-Immunology-App-a-thon`<br> Then type `activate preFDA` in anaconda prompt<br>
Type streamlit run command in anaconda prompt to see app in local environment: `streamlit run appathon.py`<br>






