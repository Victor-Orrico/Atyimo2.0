from pyspark.sql.types import *
#Define de schema of busca
schema = StructType([
        StructField("baseA", IntegerType(), True),
        StructField("dice", FloatType(), True),
        StructField("pares_candidatos", MapType(IntegerType(), FloatType(), True))
    ])

# Define schema for the next columns of the DataFrame
schema_check = StructType([
    StructField("VP", IntegerType(), True),
    StructField("VN", IntegerType(), True),
    StructField("FP", IntegerType(), True),
    StructField("FN", IntegerType(), True),
])