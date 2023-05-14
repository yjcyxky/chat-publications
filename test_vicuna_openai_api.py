import openai

openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-13b"


def test_list_models():
    model_list = openai.Model.list()
    print(model_list["data"][0]["id"])


def test_completion():
    prompt = "Once upon a time"
    completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
    print(prompt + completion.choices[0].text)


def test_embedding():
    embedding = openai.Embedding.create(model=model, input="Accordingly, retrospective analysis in 300 ME/ CFS patients indicated a decrease in IgG3 and 4, but an increase in IgG2 and IgM.[36,217,218] (continued on next page )O.A. Sukocheva, R. Maksoud, N.M. Beeraka et al. Journal of Advanced Research 40 (2022) 179–196 183 Table 1 (continued ) Author/year COVID-19 Immunologic changesME/CFS Author/year [46,96,219,220] B cells were gradually decreased (loss of transitional and follicular B cells) with increased severity of illness. SARS-CoV-2 spike-speciﬁc neutralizing antibodies, memory B cells and circulating T FH cells may be lowered in COVID-19 non- survivors. Reduced numbers of in Bcl-6+germinal centres/B cells were detected post mortem in thoracic lymph nodes and spleens in acute SARS-CoV-2 cases.B cells Dysregulated numbers of naïve, transitional, and memory B cells were reported in subsets of")
    print(len(embedding["data"][0]["embedding"]))


def test_chat_completion():
    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": "Hello! What is your name?"}]
    )
    print(completion.choices[0].message.content)


def test_chat_completion_stream():
    messages = [{"role": "user", "content": "Hello! What is your name?"}]
    res = openai.ChatCompletion.create(model=model, messages=messages, stream=True)
    for chunk in res:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    test_list_models()
    test_completion()
    test_embedding()
    test_chat_completion()
    test_chat_completion_stream()