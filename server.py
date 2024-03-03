import socket

def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected to {addr}")

        while True:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break

            print(f"Received message: {data}")

            # You can add your logic here to send messages to other connected clients

        conn.close()

if __name__ == "__main__":
    HOST = '127.0.0.1'  # Use your server IP address here
    PORT = 12345       # Choose an available port number

    start_server(HOST, PORT)
