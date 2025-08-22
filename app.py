import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
import re
import os
import sys
import warnings
import sentencepiece

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Split Bill",
    page_icon="🧾",
    layout="wide"
)

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.processor = None
    st.session_state.model = None
    st.session_state.load_error = None

@st.cache_resource
def load_models():
    try:
        with st.spinner("Loading AI models..."):
            processor = DonutProcessor.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-cord-v2",
                use_fast=True 
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-cord-v2"
            )
        return processor, model

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("This might be due to network issues or missing dependencies.")
        return None, None

def extract_receipt_data(image, processor, model):
    try:
        if not processor or not model:
            return {"error": "Models not loaded properly"}
        
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values, 
                decoder_input_ids=decoder_input_ids, 
                max_length=512,
                num_beams=1,  
            )
        
        sequence = processor.batch_decode(outputs)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "")
        sequence = sequence.replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        try:
            data = processor.token2json(sequence)
            return data
        except Exception as e:
            return {"error": f"Failed to parse receipt data: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Receipt processing failed: {str(e)}"}

def robust_parse_receipt(extracted_data):
    items = []
    subtotal = 0.0
    total = 0.0
    header = {}
    
    try:
        if not extracted_data:
            return {"error": "No data extracted from receipt"}
        
        if 'header' in extracted_data:
            header = extracted_data['header']
        elif 'receipt_info' in extracted_data:
            header = extracted_data['receipt_info']
        elif 'merchant' in extracted_data:
            header = extracted_data['merchant']
        elif 'restaurant' in extracted_data:
            header = extracted_data['restaurant']
        elif 'invoice' in extracted_data:
            header = extracted_data['invoice']
        elif 'bill' in extracted_data:
            header = extracted_data['bill']
        elif 'receipt' in extracted_data:
            header = extracted_data['receipt']
        elif 'restaurant_info' in extracted_data:
            header = extracted_data['restaurant_info']
        elif 'invoice_info' in extracted_data:
            header = extracted_data['invoice_info']
        elif 'bill_info' in extracted_data:
            header = extracted_data['bill_info']
        elif 'credit_card' in extracted_data:
            header = extracted_data['credit_card']
        elif 'payment' in extracted_data:
            header = extracted_data['payment']
        elif 'transaction' in extracted_data:
            header = extracted_data['transaction']
        elif 'order' in extracted_data:
            header = extracted_data['order']
        elif 'receipt_header' in extracted_data:
            header = extracted_data['receipt_header']
        elif 'creditcardprice' in extracted_data:
            header = extracted_data['creditcardprice']
        elif 'date' in extracted_data:
            header = extracted_data['date']
       
        menu_data = extracted_data.get('menu', [])
        if not menu_data:
            menu_data = extracted_data.get('items', [])
        if not menu_data:
            menu_data = extracted_data.get('products', [])
        if not menu_data:
            menu_data = extracted_data.get('dishes', [])
        
        for item in menu_data:
            if isinstance(item, dict):
                possible_name_keys = ['nm', 'name', 'item', 'description', 'dish', 'product_name', 'product', 'menu item', 'menu_item', 'menu_item_name', 'menu item name', 'item_name', 'item name', 'label', 'nama']

                name = next((item.get(k) for k in possible_name_keys if k in item), 'Unknown Item').strip()
                
                if name.lower() in ['restaurant', 'date', 'time', 'address', 'phone'] or 'total' in name.lower():
                    continue
                
                possible_qty_keys = ['cnt', 'quantity', 'qty', 'num', 'count', 'jumlah', 'jumlah_beli', 'jumlah beli', 'amount', 'amount_bought', 'amount bought']
                qty_str = str(next((item.get(k) for k in possible_qty_keys if k in item), '1'))
                try:
                    qty = float(qty_str.replace(',', ''))
                except:
                    qty = 1.0
                
                possible_price_keys = ['price', 'unit_price', 'uprice', 'price_each', 'unit_price_each', 'unit_price_per_item', 'item_price', 'item price', 'cost', 'cost_each', 'cost_per_item', 'harga', 'harga_satuan', 'harga satuan']
                price_str = str(next((item.get(k) for k in possible_price_keys if k in item), '0'))
                try:
                    price = float(price_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('¥', '').replace('IDR', ''))

                except:
                    price = 0.0
                
                possible_total_keys = ['sub_total', 'total', 'subtotal', 'item_total', 'line_total', 'net subtotal', 'item total', 'line total', 'net_total', 'net total', 'total_price', 'total price', 'amount', 'amount_due', 'due_amount', 'due amount']
                total_str = str(next((item.get(k) for k in possible_total_keys if k in item), qty * price))
                try:
                    item_total = float(total_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('¥', '').replace('IDR', ''))
                except:
                    item_total = qty * price
                
                if price == 0 and item_total > 0 and qty > 0:
                    price = item_total / qty
                
                if item_total <= 0 and qty == 1 and price == 0:
                    continue
                
                items.append({
                    'name': name,
                    'qty': max(qty, 1.0),
                    'price': max(price, 0.0),
                    'total': max(item_total, 0.0)
                })
        
        sub_total_data = extracted_data.get('sub_total', extracted_data.get('subtotal', {}))
        if isinstance(sub_total_data, dict):
            subtotal_str = str(sub_total_data.get('sub_total_price', sub_total_data.get('subtotal', '0')))
            try:
                subtotal = float(subtotal_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('¥', '').replace('IDR', ''))
            except:
                pass
        
        total_data = extracted_data.get('total', {})
        if isinstance(total_data, dict):
            possible_total_keys = ['total_price', 'total', 'amount', 'grand_total']
            total_str = str(next((total_data.get(k) for k in possible_total_keys if k in total_data), '0'))
            try:
                total = float(total_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('¥', '').replace('IDR', ''))
            except:
                total = 0.0
        
        calculated_subtotal = sum(item['total'] for item in items)
        if subtotal == 0.0 or subtotal < calculated_subtotal:
            subtotal = calculated_subtotal
        
        if subtotal > total and total > 0:
            subtotal = total
        
        additional_fees = max(total - subtotal, 0.0)
        
        return {
            'header': header,
            'items': items,
            'subtotal': subtotal,
            'total': total,
            'additional_fees': additional_fees
        }
        
    except Exception as e:
        return {"error": f"Failed to parse receipt: {str(e)}"}

st.title("Smart Split Bill Prototype")
st.markdown("By: Fauzi budi w")

if not st.session_state.model_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Load AI Models", type="primary", use_container_width=True):
            processor, model = load_models()
            if processor and model:
                st.session_state.processor = processor
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Models loaded successfully!")
                st.rerun()
            else:
                st.session_state.load_error = True

if st.session_state.load_error and not st.session_state.model_loaded:
    st.error("❌ Failed to load AI models. Please check your internet connection and dependencies.")
    st.info("Required packages: transformers, torch, sentencepiece, pillow")
    st.stop()

if st.session_state.model_loaded:
    uploaded_file = st.file_uploader(
        "Upload Receipt Image (JPG/PNG)", 
        type=["jpg", "png", "jpeg", "JPG"],
        help="Upload a clear photo of your receipt"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Receipt", use_container_width=True)
            
            with st.spinner("Analyzing receipt..."):
                extracted_data = extract_receipt_data(image, st.session_state.processor, st.session_state.model)
            
            if "error" in extracted_data:
                st.error(f"❌ {extracted_data['error']}")
                st.stop()
            
            with col2:
                st.subheader("Extracted Raw Data")
                st.json(extracted_data)
            
            parsed_data = robust_parse_receipt(extracted_data)
            
            if "error" in parsed_data:
                st.error(f"❌ {parsed_data['error']}")
                st.stop()
            
            header = parsed_data['header']
            items = parsed_data['items']
            subtotal = parsed_data['subtotal']
            total = parsed_data['total']
            additional_fees = parsed_data['additional_fees']
            
            if header:
                st.subheader("Receipt Header")
                for key, value in header.items():
                    st.write(f"{key.capitalize()}: {value}")
            
            st.divider()
            st.subheader("Bill Summary")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Subtotal", f"${subtotal:.2f}")
            with cols[1]:
                st.metric("Additional Fees", f"${additional_fees:.2f}")
            with cols[2]:
                st.metric("Total", f"${total:.2f}")
            
            st.divider()
            st.subheader("👥 Split the Bill")
            
            names_input = st.text_input(
                "Enter names (comma-separated)",
                placeholder="Zoro, Batman, Tony",
                help="Enter names of people splitting the bill"
            )
            
            if names_input and items:
                names = [n.strip() for n in names_input.split(",") if n.strip()]
                
                if len(names) > 0:
                    st.subheader("🛒 Assign Items")
                    
                    assignments = {}
                    for idx, item in enumerate(items):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            selected_person = st.selectbox(
                                f"{item['name']} (${item['total']:.2f})",
                                names,
                                key=f"item_{idx}"
                            )
                        with col2:
                            st.write("")  
                            st.write(f"Qty: {item['qty']}")
                        
                        assignments[item['name']] = {
                            'person': selected_person,
                            'amount': item['total']
                        }
                    
                    total_additional = additional_fees
                    
                    fee_split_method = st.radio(
                        "How to split the bill?",
                        ["Equal split", "Proportional to items"]
                    )
                    
                    if fee_split_method == "Equal split":
                        total_bill = total  
                        equal_amount = total_bill / len(names)
                        person_totals = {name: equal_amount for name in names}
                    else:
                        person_totals = {name: 0.0 for name in names}
                        for assign in assignments.values():
                            person_totals[assign['person']] += assign['amount']
                        
                        if total_additional > 0:
                            total_base_amount = sum(person_totals.values())
                            if total_base_amount > 0:
                                for name in names:
                                    proportion = person_totals[name] / total_base_amount
                                    person_totals[name] += total_additional * proportion
                            else:
                                additional_per_person = total_additional / len(names)
                                for name in names:
                                    person_totals[name] += additional_per_person
                    
                    st.divider()
                    st.subheader("💳 Final Split")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for name, amount in person_totals.items():
                            st.metric(name, f"${amount:.2f}")
                    
                    with col2:
                        calculated_total = sum(person_totals.values())
                        st.metric("Calculated Total", f"${calculated_total:.2f}")
                        st.metric("Original Total", f"${total:.2f}")
                        
                        if abs(calculated_total - total) < 0.01:
                            st.success("✅ Split verified!")
                        else:
                            st.error("❌ Split doesn't match total")
                    
                    results = {
                        "original_total": total,
                        "split_details": {
                            name: round(amount, 2) 
                            for name, amount in person_totals.items()
                        }
                    }
                    
                    st.download_button(
                        label="📥 Download Split Results",
                        data=json.dumps(results, indent=2),
                        file_name="bill_split.json",
                        mime="application/json"
                    )
                    
                else:
                    st.warning("Please enter at least one name")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure the image is clear and contains a valid receipt")
else:

    st.info("Click 'Load AI Models' above to get started!")
