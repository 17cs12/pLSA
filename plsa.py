
words = []
for i in documents:
   words += i
words = list(set(words))
number_of_words = len(words)
bag_of_words = np.zeros((number_of_documents, number_of_words))
for i in range(number_of_documents):
   for j in range(number_of_words):
       bag_of_words[i, j] = documents[i].count(words[j])
K = 3
def calculate_pzdw(pwz, pzd):
   number_of_words, K = pwz.shape
   number_of_documents = pzd.shape[1]
   pzdw = np.zeros((K, number_of_words, number_of_documents))
   for i in range(number_of_documents):
       pzdw[:, :, i] = (pwz * pzd[:, i]).T
   for i in range(number_of_documents):
       denom = pzdw[:, :, i].sum(axis=0)
       pzdw[:, :, i] /= denom
   return pzdw
def calculate_pwz(pzdw, bag_of_words):
   K, number_of_words, number_of_documents = pzdw.shape
   pwz = np.zeros((number_of_words, K))
   for j in range(number_of_words):
       pwz[j] = (bag_of_words[:, j] * pzdw[:, j, :]).sum(axis=1)
   for k in range(K):
       denom = (bag_of_words * pzdw[k].T).sum()
       pwz[:, k] /= denom
   return pwz
def calculate_pzd(pzdw, bag_of_words):
   K, number_of_words, number_of_documents = pzdw.shape
   pzd = np.zeros((K, number_of_documents))
   for k in range(K):
       pzd[k] = (bag_of_words * pzdw[k].T).sum(axis=1)
   for k in range(K):
       pzd[k] /= bag_of_words.sum(axis=1)
   return pzd
def plsa(bag_of_words, K, number_of_iterations=10, epsilon=0.0001):
   number_of_documents, number_of_words = bag_of_words.shape
   pwz = np.random.rand(number_of_words, K)
   pzd = np.random.rand(K, number_of_documents)
   normalize = pwz.sum(axis=0)
   pwz /= normalize
   normalize = pzd.sum(axis=0)
   normalize.shape = (1, number_of_documents)
   pzd /= normalize
   for i in range(number_of_iterations):
       last_pwz = np.copy(pwz)
       pzdw = calculate_pzdw(pwz, pzd)
       pwz = calculate_pwz(pzdw, bag_of_words)
       pzd = calculate_pzd(pzdw, bag_of_words)
       pwz_change = ((last_pwz - pwz) ** 2).sum()
       if pwz_change < epsilon:
           break
   return pwz, pzd
pwz, pzd = plsa(bag_of_words, K, 1000)
for k in range(K):
   print("class: ", k)
   terms = sorted(enumerate(pwz[:, k]),
   ...  key=lambda x: x[1], reverse=True)
   for term_id, score in terms[:20]:
       print("{word:25}{score:45}".format(score=str(score),
       ...  word=words[term_id]))
