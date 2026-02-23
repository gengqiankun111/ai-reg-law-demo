import os

# 优先设置：使用本地缓存，避免每次请求 modules.json
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.abspath("./model_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("./model_cache"))
os.environ.setdefault("HF_HOME", os.path.abspath("./model_cache"))

import chromadb
from chromadb.utils import embedding_functions

# 基础配置（与 config 保持一致，便于与 ingest_docs/rag_query 共用向量库）
from config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
)

def _local_embedding_model_path():
    """若 model_cache 中已有该模型，返回本地路径，加载时不访问外网。"""
    cache_dir = os.path.abspath("./model_cache")
    local_name = EMBEDDING_MODEL.replace("/", "_")
    path = os.path.join(cache_dir, "sentence_transformers", local_name)
    if os.path.isdir(path):
        return path
    return None

def _get_embedding_function():
    """优先使用本地缓存路径，避免每次下载 modules.json。"""
    local_path = _local_embedding_model_path()
    if local_path:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=local_path,
            device="cpu",
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        cache_folder=os.path.abspath("./model_cache"),
    )

DEFAULT_EMBEDDING_FUNCTION = _get_embedding_function()

def init_chroma(clear_old_data=False):
    """初始化 Chroma 客户端和集合（持久化模式）
    :param clear_old_data: 为 True 时会删除 laws_collection 再重建，会覆盖 ingest_docs 写入的向量；默认 False 与 ingest 数据共存。
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if clear_old_data:
        try:
            client.delete_collection("laws_collection")
        except:
            pass  # 集合不存在时忽略

    # 获取/创建集合（检索时直接获取已有集合，不删除）
    collection = client.get_or_create_collection(
        name="laws_collection",
        embedding_function=DEFAULT_EMBEDDING_FUNCTION,
        metadata={"description": "国家法律法规纯文本条文"}
    )
    return client, collection


def load_law_texts():
    """加载真实的法律条文纯文本（扩充版：含民法典/刑法/劳动合同法/行政复议法/民事诉讼法）"""
    law_texts = [
        {
            "id": "law_001",
            "title": "中华人民共和国民法典（核心条款）",
            "content": """
第一百八十八条 向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。
诉讼时效期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。法律另有规定的，依照其规定。但是，自权利受到损害之日起超过二十年的，人民法院不予保护，有特殊情况的，人民法院可以根据权利人的申请决定延长。

第五百零九条 当事人应当按照约定全面履行自己的义务。
当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。
当事人在履行合同过程中，应当避免浪费资源、污染环境和破坏生态。

第六百六十七条 借款合同是借款人向贷款人借款，到期返还借款并支付利息的合同。
第六百六十八条 借款合同应当采用书面形式，但是自然人之间借款另有约定的除外。
借款合同的内容一般包括借款种类、币种、用途、数额、利率、期限和还款方式等条款。

第一千一百七十九条 侵害他人造成人身损害的，应当赔偿医疗费、护理费、交通费、营养费、住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。造成残疾的，还应当赔偿辅助器具费和残疾赔偿金；造成死亡的，还应当赔偿丧葬费和死亡赔偿金。
            """,
            "source": "国家法律法规数据库",
            "publish_date": "2021-01-01"
        },
        {
            "id": "law_002",
            "title": "中华人民共和国刑法（核心条款）",
            "content": """
第二百六十四条 盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。

第二百六十六条 诈骗公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。本法另有规定的，依照规定。

第二百三十四条 故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。
犯前款罪，致人重伤的，处三年以上十年以下有期徒刑；致人死亡或者以特别残忍手段致人重伤造成严重残疾的，处十年以上有期徒刑、无期徒刑或者死刑。本法另有规定的，依照规定。

第三百八十二条 国家工作人员利用职务上的便利，侵吞、窃取、骗取或者以其他手段非法占有公共财物的，是贪污罪。
受国家机关、国有公司、企业、事业单位、人民团体委托管理、经营国有财产的人员，利用职务上的便利，侵吞、窃取、骗取或者以其他手段非法占有国有财物的，以贪污论。
与前两款所列人员勾结，伙同贪污的，以共犯论处。
            """,
            "source": "国家法律法规数据库",
            "publish_date": "2021-03-01"
        },
        {
            "id": "law_003",
            "title": "中华人民共和国劳动合同法（核心条款）",
            "content": """
第十条 建立劳动关系，应当订立书面劳动合同。
已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。
用人单位与劳动者在用工前订立劳动合同的，劳动关系自用工之日起建立。

第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：
（一）未按照劳动合同约定提供劳动保护或者劳动条件的；
（二）未及时足额支付劳动报酬的；
（三）未依法为劳动者缴纳社会保险费的；
（四）用人单位的规章制度违反法律、法规的规定，损害劳动者权益的；
（五）因本法第二十六条第一款规定的情形致使劳动合同无效的；
（六）法律、行政法规规定劳动者可以解除劳动合同的其他情形。
用人单位以暴力、威胁或者非法限制人身自由的手段强迫劳动者劳动的，或者用人单位违章指挥、强令冒险作业危及劳动者人身安全的，劳动者可以立即解除劳动合同，不需事先告知用人单位。

第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。
劳动者月工资高于用人单位所在直辖市、设区的市级人民政府公布的本地区上年度职工月平均工资三倍的，向其支付经济补偿的标准按职工月平均工资三倍的数额支付，向其支付经济补偿的年限最高不超过十二年。
本条所称月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。
            """,
            "source": "国家法律法规数据库",
            "publish_date": "2013-07-01"
        },
        {
            "id": "law_004",
            "title": "中华人民共和国行政复议法（核心条款）",
            "content": """
第六条 有下列情形之一的，公民、法人或者其他组织可以依照本法申请行政复议：
（一）对行政机关作出的警告、罚款、没收违法所得、没收非法财物、责令停产停业、暂扣或者吊销许可证、暂扣或者吊销执照、行政拘留等行政处罚决定不服的；
（二）对行政机关作出的限制人身自由或者查封、扣押、冻结财产等行政强制措施决定不服的；
（三）对行政机关作出的有关许可证、执照、资质证、资格证等证书变更、中止、撤销的决定不服的；
（四）对行政机关作出的关于确认土地、矿藏、水流、森林、山岭、草原、荒地、滩涂、海域等自然资源的所有权或者使用权的决定不服的；
（五）认为行政机关侵犯合法的经营自主权的；
（六）认为行政机关变更或者废止农业承包合同，侵犯其合法权益的；
（七）认为行政机关违法集资、征收财物、摊派费用或者违法要求履行其他义务的；
（八）认为符合法定条件，申请行政机关颁发许可证、执照、资质证、资格证等证书，或者申请行政机关审批、登记有关事项，行政机关没有依法办理的；
（九）申请行政机关履行保护人身权利、财产权利、受教育权利的法定职责，行政机关没有依法履行的；
（十）申请行政机关依法发放抚恤金、社会保险金或者最低生活保障费，行政机关没有依法发放的；
（十一）认为行政机关的其他具体行政行为侵犯其合法权益的。

第九条 公民、法人或者其他组织认为具体行政行为侵犯其合法权益的，可以自知道该具体行政行为之日起六十日内提出行政复议申请；但是法律规定的申请期限超过六十日的除外。
因不可抗力或者其他正当理由耽误法定申请期限的，申请期限自障碍消除之日起继续计算。
            """,
            "source": "国家法律法规数据库",
            "publish_date": "2024-01-01"
        },
        {
            "id": "law_005",
            "title": "中华人民共和国民事诉讼法（核心条款）",
            "content": """
第十八条 基层人民法院管辖第一审民事案件，但本法另有规定的除外。
第十九条 中级人民法院管辖下列第一审民事案件：
（一）重大涉外案件；
（二）在本辖区有重大影响的案件；
（三）最高人民法院确定由中级人民法院管辖的案件。

第六十四条 当事人对自己提出的主张，有责任提供证据。
当事人及其诉讼代理人因客观原因不能自行收集的证据，或者人民法院认为审理案件需要的证据，人民法院应当调查收集。
人民法院应当按照法定程序，全面地、客观地审查核实证据。

第一百二十二条 起诉必须符合下列条件：
（一）原告是与本案有直接利害关系的公民、法人和其他组织；
（二）有明确的被告；
（三）有具体的诉讼请求和事实、理由；
（四）属于人民法院受理民事诉讼的范围和受诉人民法院管辖。
            """,
            "source": "国家法律法规数据库",
            "publish_date": "2024-01-01"
        }
    ]
    return law_texts

def save_to_chroma(clear_old_data=False):
    """将真实法律文本存入 Chroma。默认不清空，与 ingest_docs 写入的向量共存；需仅保留爬虫数据时传 clear_old_data=True。"""
    client, collection = init_chroma(clear_old_data=clear_old_data)
    law_texts = load_law_texts()

    # 提取批量插入的数据
    ids = [item["id"] for item in law_texts]
    documents = [item["content"].strip() for item in law_texts]
    metadatas = [
        {
            "title": item["title"],
            "source": item["source"],
            "publish_date": item["publish_date"]
        } for item in law_texts
    ]

    # 批量插入
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[OK] 成功插入 {len(ids)} 条真实法律条文到 Chroma")
    # 返回客户端和集合，供检索使用（避免重复初始化）
    return client, collection


def test_rag_retrieval(client, collection, query="民法典 诉讼时效 三年", top_k=2):
    """测试 RAG 检索（使用已初始化的客户端/集合，避免清空数据）"""
    # 向量检索 TopK（直接使用传入的集合，不再重新初始化）
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print("\n=== RAG 精准检索结果 ===")
    if not results["documents"][0]:
        print("[--] 无匹配结果")
        return

    for i in range(min(top_k, len(results["documents"][0]))):
        meta = results["metadatas"][0][i]
        title = meta.get("title") or meta.get("source_file") or meta.get("source") or "未知来源"
        content = results["documents"][0][i]
        score = results["distances"][0][i]  # 分数越小，相似度越高

        print(f"\n{i + 1}. 法律名称：{title}")
        print(f"   相似度分数：{score:.4f}（越小越相似）")
        print(f"   匹配条文：\n{content}")


if __name__ == "__main__":
    # 第一步：清空旧数据，存入真实法律文本（返回客户端/集合）
    client, collection = save_to_chroma()
    # 第二步：测试精准检索（传入已有的客户端/集合，不重新初始化）
    test_rag_retrieval(client, collection, query="民法典 合同 履行义务")
    # 可更换查询词测试
    # test_rag_retrieval(client, collection, query="刑法 盗窃罪 罚金")