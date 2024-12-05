import pandas as pd
import os


def create_separate_txt_files(input_csv_path, output_dir):
    # Arabic column name mapping dictionary
    column_mapping = {
        'Clause Reference (Clause Number - Policy Name)': 'المرجع (رقم البند - اسم السياسة)',
        'Clause Content': 'محتوى البند',
        'Clause Number': 'رقم البند',
        'Policy Name': 'اسم السياسة',
        'Issuing Entity': 'الجهة المصدرة'
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Try reading CSV with UTF-8 encoding
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try cp1256 encoding (common for Arabic text)
        df = pd.read_csv(input_csv_path, encoding='cp1256')

    # Process each row and create a separate txt file
    for index, row in df.iterrows():
        # Initialize output lines for current record
        output_lines = []

        # Process each column in the row
        for original_col in df.columns:
            # Get Arabic column name from mapping, use original if not found
            arabic_col = column_mapping.get(original_col, original_col)
            value = row[original_col]
            # Add formatted line with Arabic column name
            output_lines.append(f"{arabic_col}: {value}")

        # Generate unique filename for each record
        filename = f"record_{index + 1}.txt"
        file_path = os.path.join(output_dir, filename)

        # Write content to individual text file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

    print(f"Generated {len(df)} text files in directory: {output_dir}")


# Example usage
if __name__ == "__main__":
    input_csv = "data.csv"  # Replace with your CSV file path
    output_directory = "indexing/input"  # Replace with desired output directory
    create_separate_txt_files(input_csv, output_directory)