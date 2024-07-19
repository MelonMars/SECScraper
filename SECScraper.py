import networkx as nx
import numpy as np
import requests, shutil, os, yaml, csv, json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup as bs4
import pandas as pd
import networkx as netx
from pyvis.network import Network
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from scipy.stats import norm

heads = {'Host': 'www.sec.gov', 'Connection': 'close',
         'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
         }


def download(year):
    for qtr in range(1):
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr+1}/master.idx"
        response = requests.get(url, headers=heads)
        print(url)
        response.raise_for_status()
        with open(f"SECData/master{year}QTR{qtr+1}.idx", 'wb') as f:
            f.write(response.content)
        csv_lines = []
        with open(f"SECData/master{year}QTR{qtr+1}.idx", 'r') as f:
            lines = f.readlines()[9:]
            for line in lines:
                if line.__contains__("--------------------------------------------------------------------------------"):
                    pass
                else:
                    csv_lines.append(line.replace('|', ","))
        with open(f"SECData/master{year}QTR{qtr+1}.idx", "w") as f:
            f.writelines(csv_lines)

def MapCIKToTicker():
    response = requests.get("https://www.sec.gov/include/ticker.txt", headers=heads)
    response.raise_for_status()
    with open('SECData/CIK.txt', 'wb') as f:
        f.write(response.content)
    with open('SECData/CIK.txt', 'r+') as f:
        with open('SECData/tmp', 'w') as ff:
            ff.write("cik, tik\n")
            for line in f:
                parts = line.strip().split('\t')
                flipped_line = f"{parts[1]},{parts[0]}"
                ff.write(f"{flipped_line}\n")
            shutil.move("SECData/tmp", "SECData/CIK.txt")

def GetForm13F(yr):
    with open(f"SECData/master{yr}QTR1.idx", "r") as f:
        data = pd.read_csv(f, usecols=range(5))
        index_list = data.loc[data['Form Type'] == '13F-HR'].index.tolist()
        download_urls = data.loc[index_list, 'Filename'].tolist()
        for file in download_urls:
            url = f"https://www.sec.gov/Archives/{file}"
            print(url)
            response = requests.get(url, headers=heads)
            response.raise_for_status()
            file_path = f"SECData/13F-HR/{file}"
            dir = os.path.dirname(file_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(file_path, "wb") as f:
                f.write(response.content)
    dir = f"SECData/{yr}/13F-HR/"
    for root, dirs, files in os.walk(dir):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dir, file)
            shutil.move(src_path, dest_path)
    if os.path.exists(f"SECData/{yr}/13F-HR/edgar"):
        shutil.rmtree(f"SECData/{yr}/13F-HR/edgar")


def ParseForm13F(yr):
    dir = f"SECData/{yr}/13F-HR"
    relationships = []
    errs = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            with open(os.path.join(root, file), "r+") as f:
                soup = bs4(f.read(), 'lxml')
                try:
                    yaml_data = str(soup.find("acceptance-datetime"))
                    yaml_data = "".join("\n".join(yaml_data.split("\n")[1:-1])).replace("\t", "        ")
                    data = yaml.safe_load(yaml_data)
                    name = data["FILER"]["COMPANY DATA"]["COMPANY CONFORMED NAME"].replace(',','')
                    for table in soup.find_all("infotable"):
                        doc = ET.fromstring(str(table))
                        relationships.append(f"{name},{doc.find('nameofissuer').text.replace(',','')},{doc.find('shrsorprnamt/sshprnamt').text}\n")
                        print(relationships[-1])
                except:
                    errs.append(soup.find('acceptance-datetime'))
    print(len(errs), errs)
    with open(f"SECData/{yr}/relationships.csv", "w") as f:
        f.write("source,target,weight\n")
        for line in relationships:
            f.write(line)


def ParseForm13F2(yr, target):
    df = pd.read_csv(f"SECData/{yr}/relationships.csv")
    df = df[df['source'].str.contains(target, na=False)]
    df.to_csv('output.csv', index=False)


def MapCUSIPToTicker():
    mapping = {}
    with open("SECData/cnsfails202303b.txt", 'r') as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            mapping[row["CUSIP"]] = row["SYMBOL"]
    with open("SECData/CUSIP.txt", "w") as f:
        f.write("cusip,symbol\n")
        for cusip, symbol in mapping.items():
            f.write(f"{cusip},{symbol}\n")


def CikToTik(cik):
    with open("SECData/CIK.txt", 'r') as f:
        df = pd.read_csv(f)
        df = df.set_index('cik')
        return df.loc[cik, 'tik']


def MapNameToTicker():
    mapping = {}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=heads).json()
    for val in res:
        mapping[res[val]["title"]] = res[val]["ticker"]
    with open("SECData/names.txt", "w") as f:
        json.dump(mapping, f)


def NameToTick(name):
    with open("SECData/names.txt", "r") as f:
        return json.load(f)[name]


def get_targets_recursive(df, source, depth=0, max_depth=5, subset=None):
    threshold = 80
    df['source'] = df['source'].astype(str)
    if subset is None:
        subset = pd.DataFrame(columns=df.columns)  # Initialize the subset DataFrame

    if depth > max_depth:
        return subset

    # Collect targets based on the given source name
    targets = df[df['source'].apply(lambda x: fuzz.partial_ratio(x, source)) >= threshold]['target'].unique()

    # Filter out duplicates and append to the subset DataFrame
    new_subset = pd.DataFrame({'source': [source] * len(targets),
                               'target': targets}).drop_duplicates()
    subset = pd.concat([subset, new_subset], ignore_index=True)

    # Recursively call the function for each target
    for target in targets:
        subset = get_targets_recursive(df, target, depth + 1, max_depth, subset)

    return subset

def DisplayData():
    df = pd.read_csv("output.csv")
    G = netx.from_pandas_edgelist(df, source='source', target='target')
    net = Network(notebook=True, directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.show('example2.html')

def CentralityStuff(df):
    G = netx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['weight'], create_using=nx.DiGraph())
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
    eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight")
    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    with open("sorted_degree_weighted", "w") as f:
        f.write(str(sorted_degree))
        f.close()
    with open("sorted_betweenness_weighted", "w") as f:
        f.write(str(sorted_betweenness))
        f.close()
    with open("sorted_eigenvector_weighted", "w") as f:
        f.write(str(sorted_eigenvector))
        f.close()

def DiagnosticData(df):
    G = netx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph())
    print("[*] Density:", netx.density(G))
    print("[*] Avg degree connectivity:", netx.average_degree_connectivity(G))
    print("[*] Avg clustering:", netx.average_clustering(G))


def MapCompanyToShares(df):
    cumulative_weights = df.groupby('source')['weight'].sum()
    sorted_mapping = cumulative_weights.sort_values(ascending=False)
    mapping = sorted_mapping.to_dict()
    with open('mapping.json', 'w') as json_file:
        json.dump(mapping, json_file)

def RenderPNGandJPG(df):
    G = netx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['weight'], create_using=nx.DiGraph())
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(42, 42))
    nx.draw(G, pos, node_size=10, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    output_filename = "graph1.png"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)

    plt.show()




if __name__ == "__main__":
    download(2008)
