import ipaddress
import shutil
import subprocess

from langchain_core.tools import tool


def validate_ip(ip: str) -> str:
    """Accept a single IPv4 or IPv6 address."""
    return str(ipaddress.ip_address(ip.strip()))


@tool
def discover_host(ip: str) -> str:
    """Check whether an authorized target IP responds to host discovery."""
    target = validate_ip(ip)

    result = subprocess.run(
        ["nmap", "-sn", "-n", target],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        return f"Nmap failed: {result.stderr.strip()}"

    return result.stdout


@tool
def scan_common_ports(ip: str) -> str:
    """Scan the 20 most common TCP ports on one authorized IP."""
    target = validate_ip(ip)

    result = subprocess.run(
        ["nmap", "-n", "-F", target],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        return f"Nmap failed: {result.stderr.strip()}"

    return result.stdout


class NmapToolkit:
    """Organize Nmap network reconnaissance tools."""

    def get_tools(self):
        return [discover_host, scan_common_ports]


if __name__ == "__main__":
    if shutil.which("nmap") is None:
        raise SystemExit("Nmap is not installed or is not in PATH.")

    toolkit = NmapToolkit()
    tools = toolkit.get_tools()

    for item in tools:
        print(f"{item.name} => {item.description}")

    target = input("Enter an authorized target IP: ").strip()

    try:
        output = discover_host.invoke({"ip": target})
        print("\nHost discovery results:")
        print(output)

    except (ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}")
