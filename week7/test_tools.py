from tools import execute_tool

result1 = execute_tool("calculate_average", {"grades": [80, 90, 70, 100]})
print("Calculation Result:", result1)

result2 = execute_tool("search_web", {"query": "Newton's second law"})
print("Search Result:", result2)

result3 = execute_tool("delete_files", {"path": "/"})
print("Unauthorized Tool Result:", result3)