from .config import DROP_COLUMNS
from .preprocessing import preprocess_reddit_reviews, preprocess_products
from .inference import ProductComparator

reviews_file = r"E:\Product_Comparator-main\Product_Comparator-main\example_data\Comparator.reviews.json"
specifications_file = r"E:\Product_Comparator-main\Product_Comparator-main\example_data\Comparator.specifications.json"

review_data = preprocess_reddit_reviews(reviews_file)
spec_data = preprocess_products(specifications_file, DROP_COLUMNS)

id = review_data[0]["id"]
product_name = ""
specs = {}
for i in range(len(spec_data)):
    if spec_data[i]["id"] == id:
        product_name = spec_data[i]["name"]
        specs = spec_data[i]["specifications"]
        break

model = ProductComparator()
print(
    model.analyze(
        product_name=product_name,
        reviews=review_data[0]["text"],
        specifications=specs,
    )
)