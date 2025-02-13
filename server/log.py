import os
import time

def reset_log():
    with open("server/relatorio/log.txt", "w") as file:
        file.write("")
    
def log(numero_thread, mensagem):
    #abra o arquivo para adionar texto no tipo UTF-8
    with open("server/relatorio/log.txt", "a", encoding="utf-8") as file:
        # o tempo tem que estar no seguinte formato - HH:MM:SS
        tempo = time.strftime('%H:%M:%S', time.localtime())
        file.write(f"{tempo} - Thread {numero_thread}: {mensagem}\n")