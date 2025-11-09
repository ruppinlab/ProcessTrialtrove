The GitHub repository http:/github.com/ruppinlab/ProcessTrialtrove contains three programs useful for processing data from the
clinical trial repository Trialtrove and it contains one Excel file with supplementary data associated with a paper.
The code will be directly useful only for those who have a Trialtrove license.
For others, the code should be interpreted as pseudocode that clarifies certain methods we are using to analyze
Trialtrove data.

To analyze Trialtrove data in a frozen manner, one must download a set of trials with all fields as an Excel file. In general, the trials
may be in multiple disjoint Excel files. For some purposes, it useful to convert the Excel format to ASCII text, which makes it
possible to use search tools such as awk and grep on UNIX. However, the Excel files may contain many non-ASCII characters and
an assortment of whitespace characters within each cell.

Therefore, 
trialtrove_processing_wcomments.py and character_conversion_table.py
are used to do the conversion from Excel to ASCII

Usage:
python -c 'import trialtrove_processing_wcomments; trialtrove_processing_wcomments.process("input.xlsx","output.txt")'
where the file names input.xlsx and output.txt may be replaced.

For the project entitled "Outcome Differences by Sex in Oncology Clinical Trials", Ashwin wrote the
program query_sex_differences.py that uses the auxiliary file  sex_trial_keyword_list.py.
The purpose of query_sex_differences.py is to find a subset of clinical trials in which the Trialtrove curators
may have found  a comparison of males vs. females for outcomes or side effects.

Usage: python3 query_sex_differences.py

The inputs are specified in the program with lines such as:
file_1 = pd.read_csv("/data/Vegesna_Schaffer/TrialTrove/freeze2/TrialTrove_Oncology_File1_20221223.txt",sep='\t', lineterminator='\n')
and combined with the line
data = pd.concat([file_1, file_2,file_3, file_4, file_5, file_6, file_7, file_8, file_9, file_10,file_11, file_12, file_13, file_14, file_15, file_16, file_17,file_18, file_19, file_20, file_21, file_22, file_23,
file_24,file_25, file_26, file_27, file_28, file_29, file_30, file_31,file_32]) 
and anyone else using this program would need to change those lines to specify the full paths of the input files and the correct number of files to combine.

The primary output of query_sex_differences.py is an Excel file with the following columns:
Trial ID: Unique positive integer assigned by Trialtrove to each trial
Gender Term: The term that may refer to males or females
Comparison Term: The token or substring (such as "vs.) that suggests a comparison has been done
Context: The context around the sex term that may include a comparison
Column: Either Trial Notes or Trial Results (because these are the two Trailtrove columns that contain results and where comparisons by sex may be found)

Alejandro Schaffer wrote the program findpubmed.py that finds PubMed IDs and PubMedCentral IDs in the ASCII versions of Trialtrove
files, which are produced by trialtrove_processing_wcomments.py and character_conversion_table.py

Usage:
python3 findpubmed.py --input_file Trialtrove-file --output_file_full tab-delimited-output-file

The rightmost column of the tab-delimited ASCII output file has the publication identifiers as a comma-delimited list.

Both trialtrove_processing_wcomments.py and python3 query_sex_differences.py require the pandas package of Python.

The file SupplementaryTableInterventionalTrialsUsingAI.xlsx contains a compilation of interventional clinical trials that use(d) artificial intelligence (AI) in some manner.
The list was compiled in August 2024 as part of the review paper:

Tian-Gen Chang, Seongyong Park, Alejandro A. Schaffer, Peng Jiang, Eytan Ruppin.
Ten Hallmarks of AI Contributions to Precision Oncology: Current State and Future Prospects

which is currently being revised.

Contributors: Ashwin Kammula and Alejandro Schaffer

Contact: Alejandro Schaffer (alejandro.schaffer@nih.gov)
