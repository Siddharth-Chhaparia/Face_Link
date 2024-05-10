import subprocess
import re
import socket

def get_wifi_ssid():
    try:
        # Execute netsh command to get SSID
        result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
        output = result.stdout

        # Use regular expressions to find the SSID in the output
        ssid_match = re.search(r"SSID\s*:\s*(.*)", output)
        if ssid_match:
            ssid = ssid_match.group(1).strip()
            return ssid
        else:
            return None
    except Exception as e:
        print("Error fetching WiFi SSID:", e)
        return None

import netifaces

def get_third_ipv4_address():
    interfaces = netifaces.interfaces()
    ipv4_addresses = {}
    for interface in interfaces:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ipv4_addresses[interface] = addr_info['addr']
    # Get the third item from the dictionary
    third_item = list(ipv4_addresses.values())[2]
    return third_item

third_ipv4_address = get_third_ipv4_address()



if __name__ == "__main__":
    ssid = get_wifi_ssid()
    ip_address = get_third_ipv4_address()

    if ssid:
        print("SSID:", ssid)
    else:
        print("Unable to fetch SSID.")

    if ip_address:
        print("IP Address:", ip_address)
    else:
        print("Unable to fetch IP address.")
