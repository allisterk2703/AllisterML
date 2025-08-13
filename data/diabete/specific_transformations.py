def diag_to_CodeRange(diag):
    try:
        diag_int = float(diag)
        if diag_int <= 139:
            return "001-139"
        elif diag_int <= 239:
            return "140-239"
        elif diag_int <= 279:
            return "240-279"
        elif diag_int <= 289:
            return "280-289"
        elif diag_int <= 319:
            return "290-319"
        elif diag_int <= 389:
            return "320-389"
        elif diag_int <= 459:
            return "390-459"
        elif diag_int <= 519:
            return "460-519"
        elif diag_int <= 579:
            return "520-579"
        elif diag_int <= 629:
            return "580-629"
        elif diag_int <= 679:
            return "630-679"
        elif diag_int <= 709:
            return "680-709"
        elif diag_int <= 739:
            return "710-739"
        elif diag_int <= 759:
            return "740-759"
        elif diag_int <= 779:
            return "760-779"
        elif diag_int <= 799:
            return "780-799"
        elif diag_int <= 999:
            return "800-999"
    except:
        if isinstance(diag, str) and diag.startswith("V"):
            return "V01-V91"
        elif isinstance(diag, str) and diag.startswith("E"):
            return "E000-E999"
    return None


df["code_range_1"] = df["diag_1"].apply(diag_to_CodeRange)
df["code_range_2"] = df["diag_2"].apply(diag_to_CodeRange)
df["code_range_3"] = df["diag_3"].apply(diag_to_CodeRange)

df.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)  # pas de réaffectation
logging.info("Specific transformations applied on diag_1, diag_2, diag_3")

# __________________________________________________________________________________________________________

df.drop(df[df["gender"] == "Unknown/Invalid"].index, inplace=True)  # filtre en place
logging.info("Dropped rows with 'Unknown/Invalid' in 'gender' column")

# __________________________________________________________________________________________________________

bins = [0, 10, 20, 30, 40, 50, 60]
labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59"]
df["medication_category"] = pd.cut(df["num_medications"], bins=bins, labels=labels, right=False, include_lowest=True)
df.drop(columns=["num_medications"], inplace=True)
logging.info("Created 'medication_category' column")

# __________________________________________________________________________________________________________

total_number_of_visits = df.groupby("patient_nbr")["encounter_id"].count().rename("total_nb_visits")
df.insert(df.shape[1], "total_nb_visits", df["patient_nbr"].map(total_number_of_visits))
logging.info("Created 'total_nb_visits' column")

# __________________________________________________________________________________________________________
