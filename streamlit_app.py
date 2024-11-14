
import streamlit as st
st.set_page_config(layout="wide")

from tqdm import tqdm
import pickle
import screed
from scipy.special import softmax
import pandas as pd
import mmh3
import time


alphabet = ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 
            'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
ksize = 1


def change_aa(peptide, index, aa):
    return peptide[:index] + aa + peptide[index+1:]


def mutant_peptides(peptide):
    mutants = []
    for aa in alphabet:
        for i in range(len(peptide)):        
            if aa != peptide[i]:
                mutants.append( change_aa(peptide, i, aa) )
                
    return mutants 


def build_kmers(sequence):
        sequence = clear_sequence(sequence)
        kmers = []
        n_kmers = len(sequence) - ksize + 1

        for i in range(n_kmers):
            kmer = sequence[i:i + ksize]
            kmers.append(kmer)

        return kmers
    
def get_all_possible_kmers_from_alphabet():
    last = alphabet
    current = []
    for i in range(ksize-1):
        for b in alphabet:
            for l in last:
                current.append(l+b)
        last = current
        current= []
    return last
    
def clear_sequence(sequence):
    sequence = sequence.upper()
    clear_seq = ""
    for i in sequence:
        if i in alphabet:
            clear_seq += i
    return clear_seq
    
def get_kmers_frequency(sequence):
    kmers_list_sequence = build_kmers(sequence)
    dic = {i:kmers_list_sequence.count(i) for i in get_all_possible_kmers_from_alphabet()}
    keys = list(dic.keys())
    values = list(dic.values())
    total = sum(values)
    
    frequencies = {}
    
    for i in range(len(keys)):        
        frequencies[ keys[i] ] = values[i]/total
    
    return frequencies



#########################################################################################################################################


with open("melhor_modelo_SVM_k1.pkl", 'rb') as f:
    model = pickle.load(f)
f.close()

labels_model = ['Não anticâncer', 'Anticâncer'] 
 
st.markdown("""
    <style>
        button {
            height: auto;
            width: 100% !important;
            
        }
        p {
            text-align: justify
        }
    </style>
""", unsafe_allow_html=True) 
 
st.header('Predição e otimização de peptídeos anticâncer')
st.markdown("""<p>Na aba "Classificador" é possível avaliar se peptídeos tem potencial ação anticâncer. A ferramenta
 pode receber uma ou mais sequências peptídicas em formato FASTA. </p>
 <p>Na aba "Peptídeos mutantes" a ferramenta pode ser usada para prever a atividade anticâncer de um único peptídeo e para gerar seus análogos com 
 sucessivas substituições de aminoácidos em cada posição. Esse recurso ajuda o usuário a selecionar os peptídeos mutantes, em relação ao peptídeo original, que podem
 ter uma probabilidade mais alta de apresentar atividade anticâncer.</p>
 """, unsafe_allow_html=True) 


# Criando guias
guias = st.tabs(["Classificador", "Peptídeos mutantes"])

# Conteúdo das guias
with guias[0]:
    if "sequences" not in st.session_state:
        st.session_state["sequences"] = ""

    sequences_area = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequences"], height = 300)
        
    query_sequences = []
    query_labels = []

    br = st.button("Executar", type="primary")
    ex = st.button("Use um exemplo")
    cl = st.button("Limpar")

    if br:
        progress_text = "Processando ... "
        print(progress_text)
        
        start_time = time.time()
        temp = open("temp.fas", "w")
        temp.write(sequences_area.strip())
        temp.close()
        
        query_labels = []
        query_sequences =[]

        for record in screed.open("temp.fas"):
            name = record.name
            sequence = record.sequence
            
            query_labels.append(name)
            query_sequences.append(sequence)
            
        n_queries = len(query_sequences)
         
        
        my_bar1 = st.progress(0, text="")
        counter = 0
        #predictions = {}
        query_name = []
        
        predicted_class = []
        for s in range(len(query_sequences)):
            counter += 1
            my_bar1.progress(round( (counter/len(query_sequences))*100 ), text=progress_text + str(counter) + " de " + str(len(query_sequences)))
            query_dict = get_kmers_frequency(query_sequences[s])
            query_values = list(query_dict.values())
         
            yhat = model.predict([query_values])
            #predictions[query_labels[s]] = labels_model[int(yhat)]
            query_name.append(query_labels[s])
            
            predicted_class.append(labels_model[int(yhat)])
            
         
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        st.write(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos, {int(seconds)} segundos")
        
        d = {'Nome da sequência de consulta': query_name, 'Classe predita': predicted_class }        
        
        df = pd.DataFrame(data=d,index=None)
                
        st.table(df)
        

     
    example = """>peptide_nm_3 (não anticâncer)
KIPVVAAIHGACLGGGLELALACHQRV
>peptide_nm_6 (não anticâncer)
VVNLWALHHNEKEWQQPDLFMPERFLDP
>peptide_nm_13 (não anticâncer)
FEQTGGPDLTTGSGKRTKSDRVEHKHASQ
>PHub_10 (anticâncer)
VNWKKVLGKIIKVAK
>PHub_100 (anticâncer)
KLKNFAKGVAQSLLNKASCKLSGQC
>PHub_101 (anticâncer)
CKLKNFAKGVAQSLLNKASKLSGQC
>PHub_102 (anticâncer)
GLFDVVKGVLKGVGKNVAGSLLEQLKCKLSGGC

    """


    if ex:
        st.session_state["sequences"] = example
        st.rerun()  


    if cl:
        st.session_state["sequences"] = ""
        st.rerun() 
        
with guias[1]:
    if "sequence_mutant" not in st.session_state:
        st.session_state["sequence_mutant"] = ""

    sequences_area_m = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequence_mutant"], height = 100)
        
    query_sequences = []
    query_labels = []

    br_m = st.button("Criar peptídeos mutantes e realizar a classificação", type="primary")
    ex_m = st.button("Usar um exemplo")
    cl_m = st.button("Limpar o formulário")

    if br_m:
        progress_text = "Processando ... "
        print(progress_text)
        
        start_time = time.time()
        temp_m = open("temp_m.fas", "w")
        temp_m.write(sequences_area_m.strip())
        temp_m.close()
        
        query_labels = []
        query_sequences =[]

        for record in screed.open("temp_m.fas"):
            name = record.name
            sequence = record.sequence
            break
        
        query_sequences.append(sequence)
        query_sequences = query_sequences + mutant_peptides(sequence)    
            
            
            
        #query_labels.append(name)
        #query_sequences.append(sequence)
            
        n_queries = len(query_sequences)
        
        query_labels.append(name)
        for i in range(n_queries):
            query_labels.append(name+"_mutant_"+str(i+1))
         
        
        my_bar1_m = st.progress(0, text="")
        counter = 0
        #predictions = {}
        query_name = []
        query_mutant = []
        predicted_class = []
        probabilidades = []
        for s in range(len(query_sequences)):
            counter += 1
            my_bar1_m.progress(round( (counter/len(query_sequences))*100 ), text=progress_text + str(counter) + " of " + str(len(query_sequences)))
            query_dict = get_kmers_frequency(query_sequences[s])
            query_values = list(query_dict.values())
         
            yhat = model.predict([query_values])
            probabilidade = model.predict_proba([query_values])
            
            query_name.append(query_labels[s])
            query_mutant.append(query_sequences[s])
            predicted_class.append(labels_model[int(yhat)])
            
            probabilidades.append(  max(probabilidade[0]) )
         
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        st.write(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos, {int(seconds)} segundos")
        
                
        d = {'Nome da sequência de consulta': query_name, 'Sequência peptídica': query_mutant, 'Classe predita': predicted_class, 'Probabilidade': probabilidades}
        df = pd.DataFrame(data=d,index=None)
        
        
        with st.expander(":blue[**Melhores resultados**]"):
            df_pep_selvagem = df.iloc[0]
            st.write("Dados do peptídeo original (selvagem):")
            st.write("Nome: ", df_pep_selvagem["Nome da sequência de consulta"]) 
            st.write("Sequência peptídica: ", df_pep_selvagem["Sequência peptídica"])
            st.write("Classe predita: ", df_pep_selvagem["Classe predita"])
            st.write("Probabilidade: ", str(df_pep_selvagem["Probabilidade"]))
            
            
            st.write("\n\n")
            st.write("Melhores sequências mutantes encontradas:")
            result = df.loc[ df['Probabilidade'] == max(probabilidades)].loc[ df['Classe predita'] == "Anticâncer"]
            st.write(result)
        
        with st.expander(":blue[**Resultados da varredurra completa**]"):
            st.table(df)
        

     
    example_m = """>PHub_10
VNWKKVLGKIIKVAK
    """


    if ex_m:
        st.session_state["sequence_mutant"] = example_m
        st.rerun()  


    if cl_m:
        st.session_state["sequence_mutant"] = ""
        st.rerun()










 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    




    
