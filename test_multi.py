from concurrent.futures import ProcessPoolExecutor
import logging

def square(n):
    logging.info(f"Berechne Quadrat von {n}")
    return n * n

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    numbers = [1, 2, 3, 4, 5]
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(square, numbers))
    print(f"Ergebnisse: {results}")

if __name__ == '__main__':
    main()