from hepai import HRModel

model = HRModel.connect(
    name="meta/llama3-8b",
    base_url="http://localhost:4260/apiv2"
)

funcs = model.functions()  # Get all remote callable functions.
print(f"Remote callable funcs: {funcs}")

# 请求远程模型的custom_method方法
output = model.generate("sai hello", max_tokens=1000)
# assert isinstance(output, int), f"output: type: {type(output)}, {output}"
print(f"Output of custon_method: {output}, type: {type(output)}")

# # 测试流式响应
# stream = model.get_stream(stream=True)  # Note: You should set `stream=True` to get a stream.
# print(f"Output of get_stream:")
# for x in stream:
#     print(f"{x}, type: {type(x)}", flush=True)