import os
import fitz  # PyMuPDF

def convert_pdfs_to_png(input_folder, output_folder, dpi=600):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            doc = fitz.open(pdf_path)
            
            print(f"Processing: {filename}...")

            # Calculate zoom factor to achieve target DPI
            # Standard PDF resolution is 72 DPI
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render page to an image (pixmap)
                pix = page.get_pixmap(matrix=matrix)
                
                # Construct output filename: originalname_page1.png
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_page{page_num + 1}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save as PNG
                pix.save(output_path)
                print(f"  Saved: {output_filename}")

            doc.close()

    print("\nConversion complete!")

if __name__ == "__main__":
    input_dir = "D:\Downlaods\Floor Plan PDF"  
    output_dir = "D:\Downlaods\Floor Plan PNG"

    # Ensure the input directory exists for the example to warn the user
    if os.path.exists(input_dir):
        convert_pdfs_to_png(input_dir, output_dir)
    else:
        print(f"Error: Input directory '{input_dir}' does not exist. Please create it and add PDF files.")