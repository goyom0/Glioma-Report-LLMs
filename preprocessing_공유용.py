
"""
Author: Dahyoun Lee
Affiliation: Department of Biomedical Systems Informatics,
             Yonsei University College of Medicine
Email: goyom@yuhs.ac; goyom@yonsei.ac.kr

Description:
    Preprocessing pipeline for MRI radiology reports from patients with glioma.

Associated Publication:
    Suh PS, Lee D, Bang CB, et al. 
    Predicting molecular types of adult-type diffuse gliomas based on MRI reports with large language models. 
    Eur Radiol. 2026;36(5):3743-3754. doi:10.1007/s00330-025-12211-x
"""


from openpyxl import load_workbook
import time
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os


class Preprocessing():
    def __init__(self, model_name, temperature, filepath):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.filepath = filepath

        self.llm = ChatOpenAI(
            temperature=self.temperature,
            model=model_name,
            openai_api_key=self.openai_api_key
        )

    def read_cell(self, row):
        wb = load_workbook(filename = self.filepath)
        sheet = wb.active
        row_values = [cell.value for cell in sheet[row]]
        return row_values


    def data_curator(self, row):
        print('Start: ', time.strftime('%c'))

        rowdata = self.read_cell(self.filepath, row)

        ID = rowdata[0]

        if rowdata[1]==0:
            sex = 'male'
        elif rowdata[1]==1:
            sex = 'female'

        age = rowdata[2]

        text = rowdata[-1]

        patterns_to_replace = [
            r'\w*oligodendroglioma\w*', 
            r'\w*blastoma\w*',
            r'\w*astrocytoma\w*',
            r'OG',
            r'ACM',
            r'GBM'
        ]

        patterns_to_remove = [
            r'IDH[-\w]*',
            r'IDH',
            r'wt',
            r'wildtype',
            r'1p\/19q',
            r'codel\w*',
            r'mutant',
            r'PXA',
            r'high-grade', r'high grade',
            r'low-grade', r'low grade',
            r'grade [\d]',
            r'lower[-\w]*',
            r'higher[-\w]*',
            r'III', r'IV'
        ]
        combined_pattern_replace = r'|'.join(patterns_to_replace)
        combined_pattern_remove = r'|'.join(patterns_to_remove)

        text = re.sub(combined_pattern_replace, 'glioma', text, flags=re.IGNORECASE)
        text_edited = re.sub(combined_pattern_remove, '', text, flags=re.IGNORECASE)

        # DELETE the sections of clinical impression(or information), Imaging techniques, and contrast media.
        query = '''I want you to act as a Doctor, Radiologist, examining the report of brain MRI images.
        Please read the following [input] and perform the following tasks STEP BY STEP:

        1. TRANSLATE ALL Korean text into English.
        2. DELETE ALL words and phrases related to genetic information, including gene names, genetic markers, proteins, mutations, and expressions indicating GENETIC STATUS.
        3. DELETE the sections containing writer's name (or initial consonant, 3 uppercase letters like ABC), typically found at the end of the [Interpretation].
        4. REPLACE ALL the words containing 'oligodendroglioma', 'astrocytoma' or 'glioblastoma' with 'glioma', EVEN IF there are some typos(typically 1 or 2 letters). 
        (For example: 'xanthoastrocytoma' -> 'tumor', 'GBM '-> 'tumor', 'oligodendroglioma' -> 'glioma')
        5. DELETE ALL the contents related the 'grade', such as 'grade 2', 'grade IV', or 'high grade'.
        6. DELETE the sections of 'Imaging techniques' or 'Technique'.
        7. DELETE all of dates and hospital's names.
        8. DELETE all the sections starting with 'Otherwise', 'No other' or 'No definite brain atrophy', even if there's a number in front of the sentence.
        9.  If there is an 'addendum' or 'history' section, DO NOT delete it. But delete the date of that.
        10. DELETE any lines that contain only section headers without any accompanying information, such as 'Approved by:' or similar headers.
        11. DO NOT ADD, CHANGE or DELETE any other texts except the tasks above!

        Arrange the information AS CLOSELY to the original format, making NATURAL context and CORRECT grammar.
        ALWAYS give me ONLY the FINAL OUTPUT.

        This is an example: 
        """
        [input]: This patient is 49-year-old male.
            Clinical information : Seizure (onset: 2018.05.23), P53-
            Compared with previous 서울아산병원 2018-01-15 MR.
            Findings and conclusions :
            1. About 3.0 x 2.5 x 2.2 cm sized extent heterogenous enhancing mass involving the tectum and posterior aspect of the midbrain, pons with obstructive hydrocephalus.
                - diffusion restriction and increased CBV at the enhancing mass
                - no remarkable change since 2018-01-15
                ==> DDx) High grade glioma. BRAF+
                    Cannot exclude diffuse midline glioma
            2. No remarkable finding on CT angiography.
            3. Left maxillary sinusitis.
            4. Others - unremarkable.
            [CONCLUSION]
            About 3.0 x 2.5 x 2.2 cm sized extent heterogenous enhancing mass involving the tectum and posterior aspect of the midbrain, pons with obstructive hydrocephalus
            ==> DDx) High grade glioma. BRAF+
                Cannot exclude diffuse midline glioma
            KJS 
            Approved by: LEE,DA HYOUN Mar 9, 2024 2:24:29 PM
            Completed by: LEE,DA HYOUN Mar 12, 2024 11:08:57 AM


        [output]: This patient is 49-year-old male.
            Clinical information : Seizure
            Compared with previous MR.
            Findings and conclusions :
            1. About 3.0 x 2.5 x 2.2 cm sized extent heterogenous enhancing mass involving the tectum and posterior aspect of the midbrain, pons with obstructive hydrocephalus.
                - diffusion restriction and increased CBV at the enhancing mass
                ==> DDx) glioma 
                    Cannot exclude diffuse midline glioma
            3. Left maxillary sinusitis.
            [CONCLUSION]
            About 3.0 x 2.5 x 2.2 cm sized extent heterogenous enhancing mass involving the tectum and posterior aspect of the midbrain, pons with obstructive hydrocephalus
            ==> DDx) glioma. 
                Cannot exclude diffuse midline glioma
        """

        Let's proceed STEP BY STEP!

        [input]: This patient is {age}-year-old {sex}.
        {question}'''

        prompt = PromptTemplate.from_template(query)
        input_text = prompt.format(question=text_edited, age=age, sex=sex)


        print(f'Start preprocessing: {ID},', time.strftime('%c'))
        output_text = self.llm.invoke(input_text)

        label_molecular = rowdata[3]
        who_grade = rowdata[4]

        datadict = {}   # one dict for one patient
        datadict['ID'] = ID

        if '[input]:' in output_text.content:
            edit_text = output_text.content.split('[input]:')[1]
            datadict['input'] = edit_text
        else: datadict['input'] = output_text.content

        datadict['label'] = [label_molecular]

        return datadict, output_text.content


