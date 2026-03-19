from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, \
    DocumentCompressorPipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models import get_ali_clients

ali_model, ali_embeddings = get_ali_clients()

loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8", )
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100
)
documents = text_splitter.split_documents(docs)
# 使用基础检索器
retriever = Chroma.from_documents(documents, ali_embeddings).as_retriever()

print("-------------------第一种：LLMChainExtractor压缩------------------")
# 使用上下文压缩检索器
compressor = LLMChainExtractor.from_llm(ali_model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
compressed_docs1 = compression_retriever.invoke(
    "deepseek的发展历程"
)
print("-------------------第二种：LLMChainFilter压缩后--------------------------")
# LLMChainFilter 是稍微简单但更强大的过滤器
_filter = LLMChainFilter.from_llm(ali_model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs2 = compression_retriever.invoke(
    "deepseek的发展历程"
)

print("-------------------第三种：EmbeddingsFilter压缩后--------------------------")
# 对每个检索到的文档进行额外的 LLM 调用既昂贵又缓慢。
# EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项

embeddings_filter = EmbeddingsFilter(embeddings=ali_embeddings, similarity_threshold=0.6)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs3 = compression_retriever.invoke(
    "deepseek的发展历程"
)

print("-------------------第四种：组合压缩后--------------------------")
# DocumentCompressorPipeline轻松地按顺序组合多个压缩器
'''
1.首先TextSplitters可以用作文档转换器，将文档分割成更小的块，
2.然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，
该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，
进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。
一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
3.最后 EmbeddingsFilter 根据与查询的相关性进行过滤。'''
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
# EmbeddingsRedundantFilter 去除重复的文档块
redundant_filter = EmbeddingsRedundantFilter(embeddings=ali_embeddings)
# EmbeddingsFilter 过滤掉相似度小于0.6的
relevant_filter = EmbeddingsFilter(embeddings=ali_embeddings, similarity_threshold=0.6)
# 组合以上多种方式
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
# 压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs4 = compression_retriever.invoke("deepseek的发展历程")