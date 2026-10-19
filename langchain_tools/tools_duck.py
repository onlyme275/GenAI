#Build in tools for the DuckDuckGOSearchRun
from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()
results = search_tools.invoke('ishant kafle')
print(results)

#Build tools Shell
from langchain_community.tools import ShellTool

shell_tools = ShellTool()
results = shell_tools.invoke('whoami')
print(results)

