@app.route('/form_login', methods=['POST', 'GET'])

def login():
    # Get SSID and IPv4 address
    ssid = get_wifi_ssid()
    ipv4_address = get_third_ipv4_address()
    
    # Check if SSID and IPv4 address are obtained successfully
    if ssid and ipv4_address:
        if request.method == 'POST':
            if 'Click to Register' in request.form:
                return redirect('/registration')
            else:
                name1 = request.form.get('username')
                pwd = request.form.get('password')
                
                # Check if both username and password are provided
                if name1 and pwd:
                    print("SSID:", ssid)
                    print("IPv4 Address:", ipv4_address)
                    print("Username:", name1)
                    print("Password:", pwd)
                    
                    # Check if SSID, IPv4 address, username, and password match
                    if (ssid == "Oneplus R" and ipv4_address == "192.168.29.31" and 
                        name1 in database and database[name1] == pwd):
                        return render_template('home.html', name=name1)
                    else:
                        return render_template('login.html', info='Invalid credentials')
                else:
                    return render_template('login.html', info='Please provide both username and password')
    
    # If SSID or IPv4 address not obtained, display an error message
    return render_template('login.html', info='Unable to fetch SSID or IPv4 address')
