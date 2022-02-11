import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import permutations

st.set_option('deprecation.showPyplotGlobalUse', False)

from pyisemail import is_email

import networkx as nx
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [16, 8]
plt.rcParams["figure.autolayout"] = True

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


def cc(G, nodeColor, edgeColor, font_color, opacityNode, opacityEdge, typeOfGraph, df):
	corThresh=st.sidebar.slider("Adjust threshold corelation", min(df['Corelation']), max(df['Corelation']), min(df['Corelation']))
	nodeSize=st.sidebar.slider("Adjust node size", 300, 3000, 1500)		
	if typeOfGraph == "Connected Graph":
		pos = nx.shell_layout(G)
		nodes=tuple(['All']+[i for i in G.nodes()])
		node=st.selectbox("Show nodes", nodes)
		if node=='All':	
			G1=nx.from_pandas_edgelist(df[df.Corelation>=corThresh], 'Attribute 1', 'Attribute 2', edge_attr='Corelation' , create_using=nx.DiGraph() )
			widths = nx.get_edge_attributes(G1, 'Corelation')
			nx.draw(G,edgelist=G1.edges(),nodelist=G.nodes(), pos=pos, with_labels = True, edge_color = edgeColor, arrowsize=20, node_color=nodeColor, connectionstyle='arc3, rad = 0.1', node_size=nodeSize, font_color=font_color)  
			# nx.draw(G, pos, with_labels = True, edge_color = edgeColor, arrowsize=20, node_color=nodeColor)   
			nx.draw_networkx_edge_labels(G,pos, edge_labels ={(n1,n2):widths[n1,n2] for (n1,n2) in widths if widths[n1,n2] >=corThresh}, font_color='black')
			# nx.draw_networkx_edges(G,pos,
	  #                      edgelist = {(n1,n2):widths[n1,n2] for (n1,n2) in widths.keys() if widths[n1,n2] >=corThresh},
	  #                      width=list(widths.values()),
	  #                      edge_color=edgeColor,
	  #                      alpha=opacityEdge,
	  #                      arrowsize=20, arrowstyle='->')
			plt.show()
			st.pyplot()
		else:
			dfTmp=df[df['Attribute 1']==node]
			G2=nx.from_pandas_edgelist(dfTmp[dfTmp.Corelation>corThresh], 'Attribute 1', 'Attribute 2', edge_attr='Corelation' , create_using=nx.DiGraph() )
			widths = nx.get_edge_attributes(G2, 'Corelation')
			nx.draw(G2,edgelist=G2.edges(),nodelist=G2.nodes(), pos=pos, with_labels = True, edge_color = edgeColor, arrowsize=20, node_color=nodeColor, connectionstyle='arc3, rad = 0.1', node_size=nodeSize, font_color=font_color)  
			# nx.draw(G, pos, with_labels = True, edge_color = edgeColor, arrowsize=20, node_color=nodeColor)   
			nx.draw_networkx_edge_labels(G2,pos, edge_labels ={(n1,n2):widths[n1,n2] for (n1,n2) in widths if widths[n1,n2] >=corThresh}, font_color='black')			
			plt.show()
			st.pyplot()
		

	else:
		widths = nx.get_edge_attributes(G, 'Corelation')
		nodelist = G.nodes()

		plt.figure(figsize=(12,8))

		pos = nx.shell_layout(G)
		nx.draw_networkx_nodes(G,pos,
		nodelist=nodelist,
		# node_size=,
		node_color=nodeColor,
		node_size=nodeSize,
		alpha=opacityNode)
		nx.draw_networkx_edges(G,pos,
		edgelist = widths.keys(),
		width=list(i for i in widths.values()),
		edge_color=edgeColor,
		alpha=opacityEdge,arrowsize=100, arrowstyle='->')
		nx.draw_networkx_labels(G, pos=pos,
		labels=dict(zip(nodelist,nodelist)),
		font_color=font_color)
		if st.checkbox('Show Corelation values?'):
			nx.draw_networkx_edge_labels(G,pos, edge_labels =widths, font_color='black')

		plt.box(False)
		# plt.text("mm")
		plt.show()
		st.pyplot()
		_,v,_=st.columns([0.25,0.75,0.25])
		v.write("(Thickness of edge is directly proportional to corelation)")

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# curr=conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')
	c.execute('CREATE TABLE IF NOT EXISTS data(a1 TEXT, a2 TEXT, a3 INT)')
# def check_duplicate():
# 	c.execute('SEL username from usertable WHERE ')
def add_userdata(username,password):
	c.execute('SELECT username FROM userstable WHERE username=?',[username])
	res=c.fetchone()
	if res:
		st.warning("User already exists!!")
		return False
	else:
		c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
		conn.commit()
		return True

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def homepage():
	# st.markdown("<h1 style='text-align: center; color: white;'>Corelation Visualizer using Knowledge Graph</h1>")
	st.markdown("<h1 style='text-align: center; color: white;'>Corelation Visualiser using Knowledge Graph</h1>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: center; color: grey;'>A knowledge graph, also known as a semantic network, represents a network of real-world entities—i.e. objects, events, situations, or concepts—and illustrates the relationship between them. This information is usually stored in a graph database and visualized as a graph structure, prompting the term knowledge “graph.”</p><p style='text-align: center; color: grey;'>The webapp visualizes the corelation between 2 user enetered components with the help of a user-interactive knowledge graph and heat map. To use the visualiser, select the visualize option from the drop down. You can even add your own data, by creating an account! </p>", unsafe_allow_html=True)

# @st.cache(suppress_st_warning=True)
def manipulate():
	# flag=0
	# m={'A':[], 'B': [], 'C': []}
	# df=pd.DataFrame(m)	
	c1,mid,c2=st.columns([1,1,1])
	placeholder1 = c1.empty()
	placeholder2=mid.empty()
	placeholder3=c2.empty()
	direction=st.selectbox("Direction of corelation", ('-->', '<--'))
	t1 = placeholder1.text_input('Attribute 1', key=1)
	t2 = placeholder2.text_input('Attribute 2', key=2)
	corelation = placeholder3.text_input('Corelation', key=3)
	# try:
	# corelation=float(corelation)
	# st.write(type(t1))
	# if isinstance(corelation, int) or isinstance(corelation, float):
	if st.button('Submit'):
		try:
			corelation=float(corelation)
			if direction=='<--':
				t1,t2=t2,t1
			c.execute('INSERT INTO data(a1,a2, a3) VALUES (?,?,?)',(t1,t2,corelation))
			conn.commit()
			# corelation=float(corelation)
			t1 = placeholder1.text_input('text', value='', key=5)
			t2 = placeholder2.text_input('text', value='', key=6)
			corelation = placeholder3.text_input('text', value='', key=7)
		except :
				st.warning("Invalid Input")


def heat(G, df):
	m=dict()
	r=set()
	for j,i in enumerate(list(permutations(list(G.nodes()), 2))):
	    if j%len(G.nodes())==0:
	        print(1)
	        m[(i[0],i[0])]=-1
	    x= list(df.loc[(df['Attribute 1'] == i[0]) & (df['Attribute 2'] ==i[1])]['Corelation'])
	    m[i]=0 if len(x)==0 else x[0]
	m[(i[0],i[0])]=-1
	index= list(G.nodes())
	data = np.array(list(m.values()) ).reshape(len(index),len(index))
	heat_map = sns.heatmap( data, linewidth = 0.5 , annot = True,  xticklabels=index, yticklabels=index)
	# plt.title( "HeatMap using Seaborn Method" )
	plt.ylabel('Attribute 2')
	plt.xlabel('Attribute 1')
	plt.show()
	st.pyplot()

def main():
	"""Simple Login App"""

	# st.title("Simple Login App")

	menu = ["Home","Login","SignUp", "Visualize"]
	choice = st.sidebar.selectbox("",menu)

	if choice == "Home":
		# st.subheader("Home")
		homepage()
		st.sidebar.markdown("<p style='text-align: justify; color: grey;'>This is a webapp built on Streamlit to perform analysis of data based on corelation. On the backend, the app uses the sqlite3 to store user-input bivariate data, and then it generates relevent graphs using the networkx, seaborn and matplotlib libraries.</p>", unsafe_allow_html=True)
		st.sidebar.markdown("<p style='text-align: left; color: white;'>Stack Used</p>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Python</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>StreamLit</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Sqlite3</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Pandas</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Seaborn</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Networkx</li>", unsafe_allow_html=True)
		st.sidebar.markdown("<li style='text-align: left; color: grey;'>Matplotlib</li>", unsafe_allow_html=True)



	elif choice=="Visualize":
		st.markdown("<h1 style='text-align: center; color: white;'>Visualizer</h1>", unsafe_allow_html=True)
		# st.write("")
		# st.markdown("<p style='text-align: left; color: white;'>Select your choice of visualization from the given drop down.</p>", unsafe_allow_html=True)
		c.execute('SELECT * FROM data')
		data = c.fetchall()
		# st.write(data)
		# st.write(pd.DataFrame(data))
		df=pd.DataFrame(data)
		df.columns = ['Attribute 1', 'Attribute 2', 'Corelation']
		# st.write(df)
		G=nx.from_pandas_edgelist(df, 'Attribute 1', 'Attribute 2', edge_attr='Corelation', create_using=nx.DiGraph() )
		typeOfGraph=st.selectbox("Select visualization of choice", ('Connected Graph', 'With Corelation', 'Heatmap'))
		if typeOfGraph=='Heatmap':
			heat(G, df)
		else:
			col1,col2,col3=st.columns([1,1,1])
			nodeColor=col1.selectbox("Node Color", ('Green', 'Yellow', 'Blue', 'Black', 'Violet', 'Orange', 'Lightblue'))
			edgeColor=col2.selectbox("Edge Color", ('Red', 'Yellow', 'Blue', 'Black', 'Violet', 'Orange', 'Lightblue'))
			font_color=col3.selectbox("Font Color", ('Black', 'Pink', 'Red', 'Violet', 'White', 'Blue'))
			opacityNode = st.sidebar.slider('Adjust transparancy of Node',0.0, 1.0,0.1, key=9)
			opacityEdge = st.sidebar.slider('Adjust transparancy of Edge',0.0, 1.0,0.1, key=10)
			cc(G, nodeColor, edgeColor, font_color, opacityNode, opacityEdge, typeOfGraph, df)
		
		if st.checkbox('Show Raw Data'):
			# _,mm,_=st.columns([0.1,10,0.1])
			st.dataframe(df, 2000, 2000)#['Title', 'views', 'likes', 'comments', 'duration', 'date'])
			csv = convert_df(df)#['Title', 'views', 'likes', 'comments', 'duration', 'date'])

			st.download_button(
			     label="Download data as CSV",
			     data=csv,
			     file_name='large_df.csv',
			     mime='text/csv',
			 )


	elif choice == "Login":
		username = st.sidebar.text_input("Enter Email")
		password = st.sidebar.text_input("Enter Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				st.markdown("<h1 style='text-align: center; color: white;'>Welcome!!</h1>", unsafe_allow_html=True)
				st.markdown("<p style='text-align: center; color: grey;'>Enter the attributes information and corelation to add to the database.</p>", unsafe_allow_html=True)
				manipulate()
			else:
				st.sidebar.warning("Incorrect Email/Password")

	elif choice == "SignUp":
		st.markdown("<h1 style='text-align: center; color: white;'>Create New Account</h1>", unsafe_allow_html=True)
		new_user = st.text_input("Enter Email")
		new_password = st.text_input("Password",type='password')
		if st.button("Signup"):
			if is_email(new_user, check_dns=True):
				create_usertable()
				if add_userdata(new_user,make_hashes(new_password)):
					st.success("You have successfully created a valid Account")
					st.info("Go to Login Menu to login")
				# else:
				# 	pass
				# 	# st.warning("Duplicate")
			else:
				st.warning("Enter valid email")

	else:
		x,y,z=st.columns([1,1,1])
		placeholder1 = x.empty()
		placeholder2=y.empty()
		placeholder3=z.empty()
		input = placeholder1.text_input('text', key=1)
		input = placeholder2.text_input('text', key=2)
		input = placeholder3.text_input('text', key=3)
		click_clear = st.button('clear text input', key=4)
		if click_clear:
		    input = placeholder1.text_input('text', value='', key=5)
		    input = placeholder2.text_input('text', value='', key=6)
		    input = placeholder3.text_input('text', value='', key=7)

		st.write(input)



if __name__ == '__main__':
	create_usertable()
	main()
