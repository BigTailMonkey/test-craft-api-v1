import json  # 导入json模块，用于处理JSON数据
import re  # 导入正则表达式模块

from openai import OpenAI, OpenAIError  # 从openai库导入OpenAI类和OpenAIError异常
import tiktoken  # 导入tiktoken库，用于token计数
from flask import Response, jsonify, request  # 从flask库导入Response、jsonify和request对象

from flask import Blueprint  # 从flask库导入Blueprint，用于创建蓝图

from app.config import Config  # 从app包导入Config配置类
from app.decorators import query_params  # 从app包导入自定义装饰器query_params
import htmlmin  # 导入htmlmin库，用于HTML压缩

config = Config()  # 实例化配置对象
logger = Config.logger  # 获取日志记录器

DEFAULT_MODEL = "deepseek-chat"  # 默认模型名称
MODEL_SELECTION_ENABLED = False  # 是否允许模型选择
SUPPORTED_MODELS = [  # 支持的模型列表
    {
        "name": "gpt-4o-mini",  # 模型名称
        "tokens": 128000,  # 最大token数
        "label": "gpt-4o-mini (128,000 tokens)"  # 模型标签
    },
    {
        "name": "deepseek-chat",  # 模型名称
        "tokens": 128000,  # 最大token数，指定一个大于0的数即可
        "label": "deepseek-chat (Max tokens)"  # 模型标签
    },
    {
        "name": "gpt-4o",
        "tokens": 128000,
        "label": "gpt-4o (128,000 tokens)"
    },
    {
        "name": "o1-mini",
        "tokens": 128000,
        "label": "o1-mini (128,000 tokens)"
    },
    {
        "name": "o3-mini",
        "tokens": 200000,
        "label": "o3-mini (200,000 tokens)"
    },
    {
        "name": "gpt-4-turbo",
        "tokens": 128000,
        "label": "gpt-4-turbo (128,000 tokens)"
    }
]
MAX_TOKENS = 16000  # 最大token数
ERROR_INVALID_ELEMENT = "Invalid html element."  # 无效HTML元素的错误提示


def get_model_by_name(name):  # 根据模型名称获取模型配置
    for model in SUPPORTED_MODELS:  # 遍历支持的模型列表
        if model["name"] == name:  # 如果模型名称匹配
            return model  # 返回该模型配置
    return {}  # 未找到则返回空字典


def is_o1_model_or_newer(model_name):  # 判断模型是否为o1或更新的版本（如o1、o3等）
    return "o1" in model_name or "o3" in model_name  # 检查模型名称中是否包含o1或o3


def is_prompt_length_valid(prompt, model=DEFAULT_MODEL):  # 判断prompt长度是否在模型允许范围内
    try:
        encoding = tiktoken.encoding_for_model(model)  # 获取模型对应的编码器
    except KeyError:  # 如果模型名称无效
        encoding = tiktoken.encoding_for_model("gpt-4o")  # 使用gpt-4o编码器作为兜底
        logger.log_text(f"Failed to get encoding for model {model}, falling back to gpt-4", severity="WARNING")  # 记录警告日志

    num_tokens = len(encoding.encode(prompt))  # 计算prompt的token数量
    if config.ENVIRONMENT == "production":  # 如果是生产环境
        logger.log_struct(
            {"model": model, "tokens": num_tokens},  # 记录模型和token数
            severity="INFO",
        )
    selected_model = get_model_by_name(model)  # 获取模型配置
    max_tokens = selected_model.get("tokens", MAX_TOKENS)  # 获取最大token数
    return 0 < max_tokens  # 不限制token使用量
    # return num_tokens < max_tokens  # 判断token数是否小于最大值


def is_valid_html(source_code):  # 判断输入的源码是否为有效HTML元素
    pattern = "^<(\w+).*?>.*$"  # 匹配HTML标签的正则表达式
    return bool(re.match(pattern, source_code.strip(), flags=re.DOTALL))  # 使用正则判断


def parse_html(source):  # 对HTML源码进行预处理，去除<script>标签并压缩HTML
    try:
        pattern = r"<[ ]*script.*?\/[ ]*script[ ]*>"  # 匹配<script>标签的正则
        text = re.sub(
            pattern, "", source, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)  # 替换所有<script>标签为空
        )
        html = htmlmin.minify(text, remove_comments=True, remove_empty_space=True)  # 压缩HTML，去除注释和空格
    except Exception as e:  # 捕获异常
        print(f"Error parsing HTML: {e}")  # 打印错误信息
        html = source  # 出错时返回原始HTML
    return html  # 返回处理后的HTML


def call_openai_api(prompt, role, isStream, model="", key=""):  # 调用OpenAI接口，支持流式和非流式返回
    if not model:  # 如果未指定模型
        model = DEFAULT_MODEL  # 使用默认模型

    if not key:  # 如果未指定API KEY
        key = config.API_KEY  # 使用配置中的API KEY
        client = OpenAI(api_key=key, organization="org-vrjw201KSt5hgeiFuytTSaHb")  # 创建OpenAI客户端，组织ID需替换为自己的
    else:
        client = OpenAI(api_key=key)  # 创建OpenAI客户端

    if not is_prompt_length_valid(prompt, model):  # 如果prompt过长
        if config.ENVIRONMENT == "production":  # 生产环境记录日志
            logger.log_text("Prompt too large", severity="INFO")
        return jsonify({"error": "The prompt is too long."}), 413  # 返回错误响应

    print(f"Model: {model}")  # 打印当前模型

    try:
        if model == "o1-mini":  # 如果是o1-mini模型
            messages = [
                {"role": "user", "content": prompt},  # 只传递用户内容
            ]
        else:
            messages = [
                {"role": "developer", "content": role},  # 先传递开发者角色
                {"role": "user", "content": prompt},  # 再传递用户内容
            ]

        body = {
            "model": model,  # 模型名称
            "messages": messages,  # 消息内容
            "stream": isStream,  # 是否流式返回
            "user": "TestCraftUser",  # 用户标识
        }

        if not is_o1_model_or_newer(model):  # 如果不是o1或更新的模型
            body["temperature"] = 0.5  # 设置temperature参数

        response = client.chat.completions.create(**body)  # 调用OpenAI接口

        if not isStream:  # 如果不是流式返回
            print(response)  # 打印响应
            return response  # 直接返回响应

        def generate():  # 定义生成器函数
            for part in response:  # 遍历响应流
                filtered_chunk = {
                    "choices": part.model_dump().get("choices"),  # 只保留choices字段
                }
                yield f"data: {json.dumps(filtered_chunk)}\n\n".encode()  # 以SSE格式返回

        return Response(generate(), mimetype="text/event-stream")  # 返回流式响应
    except OpenAIError as e:  # 捕获OpenAI异常
        print(f"Error: {e}")  # 打印错误信息
        return jsonify({"error": str(e.message)}), e.status_code  # 返回错误响应


api = Blueprint("api", __name__)  # 创建Flask蓝图对象


@api.route("/api/ping", methods=["GET"])  # 定义/ping接口，GET方法
@query_params()  # 使用自定义装饰器
def ping():  # 健康检查接口
    return jsonify({"pong": True}), 200  # 返回pong


@api.route("/api/models", methods=["GET"])  # 定义/models接口，GET方法
def models():  # 获取支持的模型列表
    open_ai_api_key = request.args.get("open_ai_api_key", "")  # 获取请求参数中的API KEY
    if open_ai_api_key == "":  # 如果未提供API KEY
        open_ai_api_key = config.API_KEY  # 使用配置中的API KEY
        client = OpenAI(
            api_key=open_ai_api_key, organization="org-vrjw201KSt5hgeiFuytTSaHb"  # 创建OpenAI客户端
        )
    else:
        client = OpenAI(api_key=open_ai_api_key)  # 创建OpenAI客户端
    response = client.models.list()  # 获取模型列表
    models_list = response.model_dump().get("data")  # 获取模型数据
    filtered_list = [
        {"label": f"{model['label']}", "id": model["name"]}  # 构造返回的模型信息
        for model in SUPPORTED_MODELS
        if any(model["name"] == openai_model["id"] for openai_model in models_list)  # 只返回支持的模型
    ]
    response = {
        "models": filtered_list,  # 支持的模型列表
        "default_model": DEFAULT_MODEL,  # 默认模型
        "model_selection_enabled": MODEL_SELECTION_ENABLED,  # 是否允许选择模型
    }
    return response, 200  # 返回响应


@api.route("/api/generate-ideas", methods=["POST"])  # 定义/generate-ideas接口，POST方法
@query_params()  # 使用自定义装饰器
def generate_ideas(source_code, stream=True, open_ai_api_key="", model=""):  # 根据HTML元素生成测试思路
    if not is_valid_html(source_code):  # 判断HTML是否合法
        return jsonify({"error": ERROR_INVALID_ELEMENT}), 400  # 不合法返回错误

    if config.ENVIRONMENT == "production":  # 生产环境记录日志
        logger.log_struct(
            {
                "mode": "Ideas",
                "model": model,
            },
            severity="INFO",
        )  # 记录日志，包含模式和模型信息

    role = "You are a Software Test Consultant"  # 设置角色为软件测试顾问

    prompt = f"""
        Generate test ideas based on the HTML element below. Think this step by step, as a real Tester would.
        Focus on user-oriented tests that do not refer to HTML elements such as divs or classes.
        Include negative tests and creative test scenarios.
        Format the output as unordered lists, with a heading for each required list, such as Positive Tests
         or Negative Tests. Don't include any other heading.
        HTML:
        ```
        {parse_html(source_code)}
        ```

        Format the output as the following example:
        Positive Tests:
        <Idea 1>

        Negative Tests:
        <Idea 1>

        Creative Test Scenarios:
        <Idea 1>
        """  # 构造prompt，要求生成不同类型的测试思路

    return call_openai_api(prompt, role, stream, key=open_ai_api_key, model=model)  # 调用OpenAI接口获取测试思路


@api.route("/api/automate-tests", methods=["POST"])  # 定义/automate-tests接口，POST方法
@query_params()  # 使用自定义装饰器
def automate_tests(
    source_code,
    base_url,
    framework,
    language,
    pom=True,
    stream=True,
    open_ai_api_key="",
    model="",
):  # 自动生成自动化测试代码
    if not is_valid_html(source_code):  # 判断HTML是否合法
        return jsonify({"error": ERROR_INVALID_ELEMENT}), 400  # 不合法返回错误

    if config.ENVIRONMENT == "production":  # 生产环境记录日志
        logger.log_struct(
            {
                "mode": "Automate",
                "language": language,
                "framework": framework,
                "pom": pom,
                "model": model,
            },
            severity="INFO",
        )  # 记录日志，包含自动化相关参数

    role = "You are a Test Automation expert"  # 设置角色为自动化测试专家

    prompt = f"""
    Generate {framework} tests using {language} based on the html element below.
    Use {base_url} as the baseUrl. Generate as much tests as possible.
    Always try to add assertions.
    Do not include explanatory or introductory text. The output must be all {language} code.
    Format the code in a plain text format without using triple backticks.
    """  # 构造prompt，要求生成自动化测试代码

    if framework == "playwright":  # 如果框架为playwright
        prompt += """
    Use playwright/test library.
    """  # 补充playwright相关说明

    if pom:  # 如果需要生成POM结构
        prompt += """
    Create page object models and use them in the tests.
    Selectors must be encapsulated in properties. Actions must be encapsulated in methods.
    Include a comment to indicate where each file starts.
    """  # 补充POM相关说明

    prompt += f"""
    Html:
    ```
    {parse_html(source_code)}
    ```
    """  # 添加HTML内容到prompt

    return call_openai_api(prompt, role, stream, key=open_ai_api_key, model=model)  # 调用OpenAI接口获取自动化测试代码


@api.route("/api/automate-tests-ideas", methods=["POST"])  # 定义/automate-tests-ideas接口，POST方法
@query_params()  # 使用自定义装饰器
def automate_tests_ideas(
    source_code,
    base_url,
    framework,
    language,
    ideas,
    pom=True,
    stream=True,
    open_ai_api_key="",
    model="",
):  # 根据测试思路生成自动化测试代码
    if not is_valid_html(source_code):  # 判断HTML是否合法
        return jsonify({"error": ERROR_INVALID_ELEMENT}), 400  # 不合法返回错误

    if config.ENVIRONMENT == "production":  # 生产环境记录日志
        logger.log_struct(
            {
                "mode": "Automate-Ideas",
                "language": language,
                "framework": framework,
                "pom": pom,
                "model": model,
            },
            severity="INFO",
        )  # 记录日志，包含自动化相关参数

    role = "You are a Test Automation expert"  # 设置角色为自动化测试专家
    line_tab = "\n\t"  # 定义换行和缩进
    prompt = f"""
    Using the following html:

    Html:
    ```
    {parse_html(source_code)}
    ```

    Generate {framework} tests using {language} for the following Test Cases:

    TestCases:
    ```
        {line_tab.join(ideas)}
    ```

    Use {base_url} as the baseUrl.
    Always try to add assertions.
    Do not include explanatory or introductory text. The output must be all {language} code.
    Format the code in a plain text format without using triple backticks.
    """  # 构造prompt，包含测试思路和HTML内容

    if framework == "playwright":  # 如果框架为playwright
        prompt += """
    Use playwright/test library.
    """  # 补充playwright相关说明

    if pom:  # 如果需要生成POM结构
        prompt += """
    Create page object models and use them in the tests.
    Selectors must be encapsulated in properties. Actions must be encapsulated in methods.
    Include a comment to indicate where each file starts.
    """  # 补充POM相关说明

    return call_openai_api(prompt, role, stream, key=open_ai_api_key, model=model)  # 调用OpenAI接口获取自动化测试代码


@api.route("/api/check-accessibility", methods=["POST"])  # 定义/check-accessibility接口，POST方法
@query_params()  # 使用自定义装饰器
def check_accessibility(source_code, stream=True, open_ai_api_key="", model=""):  # 检查HTML元素可访问性
    if not is_valid_html(source_code):  # 判断HTML是否合法
        return jsonify({"error": ERROR_INVALID_ELEMENT}), 400  # 不合法返回错误

    if config.ENVIRONMENT == "production":  # 生产环境记录日志
        logger.log_struct(
            {
                "mode": "Ideas",
                "model": model,
             },
            severity="INFO",
        )  # 记录日志，包含模式和模型信息

    role = "You are an expert on Web Accessibility"  # 设置角色为Web可访问性专家

    prompt = f"""
        Check the HTML element below for accessibility issues according to WCAG 2.1.
        Think about this step by step. First, assess the element against each criterion.
        Then, report the result in the format specified below.
        For the criteria that cannot be assessed just by looking at the HTML, create accessibility tests.
        In the report, each criteria must be a link to the reference documentation.

        Html:
        ```
        {source_code}
        ```

        Format the output as the following example:
        - Issues
        - Conformance Level A -
        - Issue:
        - Criteria:
        - Solution:

        - Conformance Level AA -
        - Issue:
        - Criteria:
        - Solution:

        - Conformance Level AAA -
        - Issue:
        - Criteria:
        - Solution:

        - Suggested Tests
        - Test:
        - Criteria:
        - Test Details:
        """  # 构造prompt，要求检查HTML可访问性并输出分级报告

    return call_openai_api(prompt, role, stream, key=open_ai_api_key, model=model)  # 调用OpenAI接口获取可访问性报告


@api.route("/api/get-regex-for-run", methods=["POST"])  # 定义/get-regex-for-run接口，POST方法
@query_params()  # 使用自定义装饰器
def get_regex_for_run(tests, requirement, open_ai_api_key="", model=""):  # 根据测试用例和需求生成正则表达式
    role = "You are a Test Automation expert"  # 设置角色为自动化测试专家

    prompt = f"""
        I have a Mocha test framework.
        I need you to create a regular expression to include in a grep command to run tests.

        Below you will find two things:
        - A JSON with suites, each containing an array of test names.
        - A User Requirement to create the grep command

        You will have to do the following:
        1. Review each suite name. If directly related to the User Requirement, add it to the regular expression.
        2. Review each test name. If directly related to the User Requirement, add it to the regular expression.

        Only respond with the regular expression.
        If the User Requirement is not related to any suite or test, respond with ".*" to run all the tests.
        Use the format of the example response.

        Example response:
        Regex: Add User|Update User|Patch User

        JSON
        ```
        {tests}
        ```

        Requirement:
        {requirement}
        """  # 构造prompt，要求根据测试用例和需求生成正则表达式

    response = call_openai_api(prompt, role, False, key=open_ai_api_key, model=model)  # 调用OpenAI接口获取正则表达式
    return response.choices[0].message.content  # 返回正则表达式
