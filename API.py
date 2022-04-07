
from flask import *
import time
from Désordres1fichier1tableauREAL import tableauDésordres
import re

app = Flask(__name__)

class Document():
	def __init__(self, url):
		self.url = url
		self.name = ""
		self.type_document = ""
		self.desordres = []

	def rechercheDesordres(self):
		tab = tableauDésordres(self.url)

		return tab

	def add_Desordre(self, desordre):
		self.desordres.append(desordre)

	def get_Url(self):
		return self.url

	def set_Url(self, url):
		self.url = url

	def get_Name(self, name):
		return self.name

	def set_Name(self, name):
		self.name = name

	def get_TypeDocument(self, type_document):
		return self.type_document

	def set_TypeDocument(self, type_document):
		self.type_document = type_document

	def __str__(self):
		list_des = []
		for elem in self.desordres:
			list_des.append(elem)

		#print(dict_des)

		return  {
			'url': self.url,
			'name': self.name,
			'type_document': self.type_document,
			'desordres': list_des
		}

class Desordre():
	def __init__(self):
		self.url = ""
		self.description = ""

	def get_Url(self):
		return self.url

	def set_Url(self, url):
		self.url = url

	def get_Description(self):
		return self.description

	def set_Description(self,description):
		self.description = description  

	def __str__(self):
		return self.description

#url = ""
'''
@app.route('/desordres', methods=['POST'])
def url():
	url = request.form['url']

	return url
'''
@app.route('/test')
def test():
	y = 'YEAH'

	return y


@app.route('/desordres', methods=['GET'])
def desordres():
	#url = 'http://www.actis-assurances.com/downloads/sinistres/Expertise-Sinistre.pdf'
	url = request.args.get('url')
	doc = Document(url)
	#data = tableauDésordres(url)
	data = doc.rechercheDesordres()
	data = data.to_json(orient='records')
	dictdata = json.loads(data)
	
	for i in range(len(dictdata)):
		desordre = Desordre()
		desordre.set_Url(doc.get_Url())
		for key, value in dictdata[i].items():
			if key == 'Fichier':
				nm = re.search('[-._%;\w]+.pdf', value)
				name = nm[0]
				doc.set_Name(name)
			if key == 'Type de Fichier':
				doc.set_TypeDocument(value)
			if key == 'Description' :
				desordre.set_Description(value)
				doc.add_Desordre(desordre.__str__())

	return doc.__str__()

if __name__ == '__main__':
	app.run()#debug = True