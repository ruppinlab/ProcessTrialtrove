"""program to parse Trialtrove ASCII data and find PubMed ids and PMC ids"""
import os
import argparse
import io
import re

ID_LENGTH_BOUND = 30 #upper bound on number of digits in a publication identifier
INDEX_FIELD = 0
TRIAL_ID_FIELD = 1
TITLE_FIELD = 2
PHASE_FIELD = 3
STATUS_FIELD = 4
DISEASE_FIELD = 6
PRIMARY_DRUG_FIELD = 16
OTHER_DRUG_FIELD = 20
RESULTS_FIELD = 76
NOTES_FIELD = 77

pubmed_full_pattern = re.compile(r'pubmed[^\s]*\d+')
pubmed_longer_pattern = re.compile(r'pubmed.ncbi.nlm.nih.gov/\d+')
pubmed_old_pattern = re.compile(r'PubMed[^\s]*\d+')
PMC_pattern = re.compile(r'PMC\d+')
Digits_only_pattern = re.compile(r'\d+')
NCT_pattern = re.compile(r'NCT[0-9]+')

#given a string of trial codes some of which may start with NCT, return a comma-separated list of NCT codes
#The NCT codes are from ClinicalTrials.gov
def get_NCT_code(arg_trial):
       return_string = ""
       local_first_match = re.findall(NCT_pattern, arg_trial)
       if (local_first_match):
           local_num_matches = len(local_first_match)
           for local_i in range(0,local_num_matches):
               if (0 == local_i):
                   NCT_STRING = local_first_match[0]
               else:
                   NCT_STRING = NCT_STRING + "," + local_first_match[local_i]
           return_string = NCT_STRING
       return(return_string)

#given a superstring that contains a numerical PubMed or PubMed Central identifier, return the all-digits
#substring therein
def get_pubmed_digits(arg_superstring):
       local_digits_match = Digits_only_pattern.search(arg_superstring)
       local_numerical_id = local_digits_match[0]
       return(local_numerical_id)


def main():
    parser = argparse.ArgumentParser(description='Do a trial run')
    parser.add_argument("--input_file", help ="Input file with list of trials")
    parser.add_argument("--output_file_full", help ="output file for full field")
    args = parser.parse_args()
    DATA_FD = io.open(args.input_file, 'r', encoding = 'utf-8', errors='ignore')
    FULL_OUTPUT_FD = open(args.output_file_full, 'w')
    
    num_lines = 0
    for trial_line in DATA_FD:
        if (num_lines > 0):
            #   print(num_lines)
            one_trial = trial_line.rstrip(os.linesep)
            fields = one_trial.split('\t')
            pubmed_list = []
            index_of_trial = fields[INDEX_FIELD]
            code_of_trial = fields[TRIAL_ID_FIELD]
            name_of_trial = fields[TITLE_FIELD]
            phase_of_trial = fields[PHASE_FIELD]
            status_of_trial = fields[STATUS_FIELD]
            disease_of_trial = fields[DISEASE_FIELD]            
            this_results = fields[RESULTS_FIELD]
            primary_drugs = fields[PRIMARY_DRUG_FIELD]
            secondary_drugs = fields[OTHER_DRUG_FIELD]
            this_notes = fields[NOTES_FIELD]
            NCT_list = get_NCT_code(code_of_trial)
            first_match = pubmed_full_pattern.search(this_results)
            if (first_match):
                for one_pattern_match in pubmed_full_pattern.finditer(this_results):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_results[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    pubmed_list.append(numerical_id)
            second_match = pubmed_full_pattern.search(this_notes)
            if (second_match):
                for one_pattern_match in pubmed_full_pattern.finditer(this_notes):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_notes[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    if (numerical_id not in pubmed_list):
                        pubmed_list.append(numerical_id)
            third_match = pubmed_longer_pattern.search(this_results)
            if (third_match):
                for one_pattern_match in pubmed_longer_pattern.finditer(this_results):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_results[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    if (numerical_id not in pubmed_list):
                        pubmed_list.append(numerical_id)
            fourth_match = pubmed_longer_pattern.search(this_notes)
            if (fourth_match):
                for one_pattern_match in pubmed_longer_pattern.finditer(this_notes):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_notes[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    if (numerical_id not in pubmed_list):
                        pubmed_list.append(numerical_id)
            fifth_match = pubmed_old_pattern.search(this_results)
            if (fifth_match):
                for one_pattern_match in pubmed_old_pattern.finditer(this_results):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_results[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    if (numerical_id not in pubmed_list):
                        pubmed_list.append(numerical_id)
            sixth_match = pubmed_old_pattern.search(this_notes)
            if (sixth_match):
                for one_pattern_match in pubmed_old_pattern.finditer(this_notes):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_notes[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    if (numerical_id not in pubmed_list):
                        pubmed_list.append(numerical_id)                        
            seventh_match = PMC_pattern.search(this_results)
            if (seventh_match):
                for one_pattern_match in PMC_pattern.finditer(this_results):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_results[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    string_to_append = "PMC" + numerical_id
                    pubmed_list.append(string_to_append)
            eighth_match = PMC_pattern.search(this_notes)
            if (eighth_match):
                for one_pattern_match in PMC_pattern.finditer(this_notes):
                    this_end_position = one_pattern_match.end()
                    candidate_string = this_notes[this_end_position-ID_LENGTH_BOUND:this_end_position]
                    numerical_id = get_pubmed_digits(candidate_string)
                    string_to_append = "PMC" + numerical_id
                    if (string_to_append not in pubmed_list):
                        pubmed_list.append(string_to_append)
            #if (len(pubmed_list) > 0 and (status_of_trial == "Completed") and NCT_list):
            if (len(pubmed_list) > 0):            
                FULL_OUTPUT_FD.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (index_of_trial, NCT_list,phase_of_trial,status_of_trial,disease_of_trial,primary_drugs,secondary_drugs,name_of_trial))
                num_ids = len(pubmed_list)
                for i in range(0,num_ids):
                    if (i < (num_ids -1)):
                        FULL_OUTPUT_FD.write("%s," % (pubmed_list[i]))
                    else:
                        FULL_OUTPUT_FD.write("%s" % (pubmed_list[i]))
                FULL_OUTPUT_FD.write("\n")
        num_lines = num_lines+1        
    DATA_FD.close()
    FULL_OUTPUT_FD.close()

main()

