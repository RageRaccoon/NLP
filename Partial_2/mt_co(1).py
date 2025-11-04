texto1 ="jose va a reprobar la materia de pln ya que no le gusta poner atencion"
import time
import tracemalloc
import numpy as np

def indice_palabra(palabra, vec_palabras):
    for j in range(len(vec_palabras)):
        if vec_palabras[j] == palabra:
            return j

class Tokenizer:
    """ Class for tokenizing text """
    delimiter = ""
    
    """ Constructor """
    def __init__(self):
        self.delimiter = " \t\n\r\f\v" + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}"

    """ Methods """
    def verify_word(self, text:str) -> str:
        numbers = "0123456789"
        is_only_number = True
        word = ""
        for char in text:
            if char not in numbers:
                is_only_number = False
                break 

        if is_only_number:
            word = text
        else:
            for char in text:
                if char.isalpha():  # Keep letters
                    word += char
        return word
    
    def to_lowercase(self, token:list) -> list:
        for i in range(len(token)):
            for c in token[i]:
                if (c >= 'A') and (c <= 'Z'):
                    token[i] = token[i].replace(c, chr(ord(c) + 32))
        return token
    
    def remove_stopwords(self, token:list) -> list:
        stopwords = ['the', 'of', 'in', 'on', 'a', 'an', 'some', 'and', 'that', 'this', 'mi', 'es', 'a', 'lo', 'la', 'el']
        return [word for word in token if word not in stopwords]
        
        
    def tokenize(self, text: str) -> list:              
        t_init = time.time()
        tracemalloc.start()
        
        token = []
        n = len(text)
        
        i = 0
        j = i
        
        while i <= n - 1:
            if (text[i] in self.delimiter) and (text[j] in self.delimiter):
                j += 1
            elif (text[i] in self.delimiter):
                word_verified = self.verify_word(text[j:i])
                if word_verified:  
                    token.append(word_verified)
                j = i + 1
            i += 1

        if j < n:
            word_verified = self.verify_word(text[j:n])
            if word_verified:
                token.append(word_verified)

        token = self.to_lowercase(token)
        
        token = self.remove_stopwords(token)

        # print("Time:", time.time() - t_init)
        # print("Memory:", tracemalloc.get_traced_memory())
        tracemalloc.stop()
        
        return token
    
## poner eliminador de palabras unicas
def elimi_pal_repeti(token:list) -> list:
    vec_pal_unicas = []
    for i in range(len(token)):
        if token[i] not in vec_pal_unicas:
            vec_pal_unicas.append(token[i])
    return vec_pal_unicas

def matriz_concurrencia(texto:str) -> np.ndarray:
    text_toke = Tokenizer(texto1)
    vec_pal_uicas = elimi_pal_repeti(text_toke)
    n = lan(vec_pal_unicas)
    matriz_concurrencia = np.zeros((n,n))
    for i in range (len(text_toke)-1):
        pal = text_toke[i]
        pal_siguien = text_toke[i+1]
        indice_pal = indice_palabra(pal, vec_pal_unicas)
        indice_pal_siguien = indice_palabra(pal_siguien, vec_pal_unicas)


        if indice_pal == indice_pal_siguien:
            matriz_concurrencia[indice_pal][indice_pal_siguien] += 1
        else:
            matriz_concurrencia[indice_pal_siguien][indice_pal] += 1  
    return matriz_concurrencia

def similitud_coseno(palabra1, palabra2, matriz_concurrencia, texto:str) -> float:
    text_toke = Tokenizer(texto1)
    vec_pal_uicas = elimi_pal_repeti(text_toke)
    indice_pal1 = indice_palabra(palabra1, vec_pal_uicas)
    indice_pal2 = indice_palabra(palabra2, vec_pal_uicas)
    
    vec1 = matriz_concurrencia[indice_pal1]
    vec2 = matriz_concurrencia[indice_pal2]
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity
    
# Grado de correlacion entre dos palabras, se mide al sumar todos los valores de cada vector asociados a una palabra
def grado_correlacion(palabra, matriz_concurrencia, texto:str) -> float:
    text_toke = Tokenizer(texto1)
    vec_pal_uicas = elimi_pal_repeti(text_toke)
    vector_concurrencia_palabras_unicas = [] * len(vec_pal_uicas)

    for palabra in vec_pal_uicas:
        indice_pal = indice_palabra(palabra, vec_pal_uicas)
        vec = matriz_concurrencia[indice_pal]
        vector_concurrencia_palabras_unicas[indice_pal] = np.sum(vec)

    vector_ordenado_concurrencia_palabras_unicas = np.argsort(-vector_concurrencia_palabras_unicas, kind='stable')

    vector_ordenado_concurrencia = vector_concurrencia_palabras_unicas[vector_ordenado_concurrencia_palabras_unicas]

    return vector_ordenado_concurrencia
    
def pca(matriz_concurrencia:np.ndarray) -> np.ndarray:
    matriz_centrada = matriz_concurrencia - np.mean(matriz_concurrencia, axis=0)
    U, S, Vt = np.linalg.svd(matriz_centrada)
    return Vt.T[:, :2]

def main():
    text_toke = Tokenizer()
    tokens = text_toke.tokenize(texto1)
    print("Tokens:", tokens)
    
    vec_pal_unicas = elimi_pal_repeti(tokens)
    print("Palabras unicas:", vec_pal_unicas)
    
    matriz_concur = matriz_concurrencia(texto1)
    print("Matriz de concurrencia:\n", matriz_concur)
    
    palabra1 = "jose"
    palabra2 = "materia"
    similitud = similitud_coseno(palabra1, palabra2, matriz_concur, texto1)
    print(f"Similitud coseno entre '{palabra1}' y '{palabra2}':", similitud)
    
    grado_corr = grado_correlacion(palabra1, matriz_concur, texto1)
    print(f"Grado de correlacion para '{palabra1}':", grado_corr)
    
    pca_result = pca(matriz_concur)
    print("PCA Result:\n", pca_result)

if __name__ == "__main__":
    main()


