import main

while True:
    text = input('brahmi > ')
    result, error = main.interpret('<stdin>', text)

    if error:
        print(error)
    else:
        print(result)
