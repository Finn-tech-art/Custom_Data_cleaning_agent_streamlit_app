import streamlit as st
import pandas as pd
import time
import random
import re
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
import os
import io

# Optional Groq client
try:
    from groq import Groq
except Exception:
    Groq = None

# -------------------------
# STREAMLIT APP CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="UK Company City Classifier",
    page_icon="🏙️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f4fd;
        border: 1px solid #b8daff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CONFIGURATION (Same as your existing code)
# -------------------------
COMPANIES_HOUSE_KEY = os.getenv("COMPANIES_HOUSE_KEY") or "f47ac6ff-e38d-4a78-86b6-e8576d615ac3"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "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"
GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

DELAY_MIN = 0.25
DELAY_MAX = 0.6
CHECKPOINT_EVERY = 100

LANDMARK_KEYWORDS = [
    "Square","Avenue","Road","House","Street","Lane","Drive","Court","Terrace","Place",
    "Gardens","Row","Close","Crescent","Way","Hill","Parade","Mews","Broadway","Quay","Station",
    "Building","Mount"
]

CITY_WHITELIST = [
    "Bath","Birmingham","Bradford","Brighton and hove","Bristol","Cambridge","Canterbury",
    "Carlisle","Chelmsford","Chester","Chichester","Colchester","Coventry","Derby","Doncaster",
    "Durham","Ely","Exeter","Gloucester","Hereford","Kingston upon hull","Lancaster","Leeds",
    "Leicester","Lichfield","Lincoln","Liverpool","London","Manchester","Milton keynes",
    "Newcastle upon tyne","Norwich","Nottingham","Oxford","Peterborough","Plymouth","Portsmouth",
    "Preston","Ripon","Salford","Salisbury","Sheffield","Southampton","Southend on sea",
    "St albans","Stoke on trent","Sunderland","Truro","Wakefield","Wells","Westminster",
    "Winchester","Wolverhampton","Worcester","York","Aberdeen","Dundee","Dunfermline","Edinburgh",
    "Glasgow","Inverness","Stirling","Bangor","Cardiff","Newport","St asaph","St davids","Wrexham",
    "Armagh","Belfast","Derry","Lisburn","Newry","Bangor","Coleraine","Ballymena","Londonderry"
]

CH_BASE = "https://api.company-information.service.gov.uk"
GOVUK_COMPANY_BASE = "https://find-and-update.company-information.service.gov.uk/company"

# -------------------------
# UTILITY FUNCTIONS (Same as your existing code)
# -------------------------
def log(msg: str):
    """Timestamped console log."""
    print(f"[{datetime.utcnow().isoformat()}] {msg}")

def polite_sleep():
    """Small randomized sleep to avoid overloading APIs."""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b", re.I)

def extract_postcode(text: str) -> str:
    """Return first UK postcode found or empty string."""
    if not text:
        return ""
    m = POSTCODE_RE.search(text)
    return m.group(1).upper().strip() if m else ""

def first_landmark_address(full_address: str) -> str:
    """Return substring of full_address up to and including the first landmark keyword."""
    if not full_address or not isinstance(full_address, str):
        return ""
    earliest_end = None
    for kw in LANDMARK_KEYWORDS:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.I)
        m = pattern.search(full_address)
        if m:
            end = m.end()
            if earliest_end is None or end < earliest_end:
                earliest_end = end
    if earliest_end:
        return full_address[:earliest_end].strip()
    return full_address.split(",")[0].strip()

def ch_auth():
    return (COMPANIES_HOUSE_KEY, "")

def ch_search_company(company_name: str, max_results: int = 5):
    try:
        url = f"{CH_BASE}/search/companies"
        params = {"q": company_name, "items_per_page": max_results}
        r = requests.get(url, params=params, auth=ch_auth(), timeout=15)
        if r.status_code == 200:
            return r.json().get("items", [])
        else:
            return []
    except Exception as e:
        return None

def ch_get_company_profile(company_number: str):
    try:
        url = f"{CH_BASE}/company/{company_number}"
        r = requests.get(url, auth=ch_auth(), timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception as e:
        return None

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_govuk_soup(company_number: str):
    try:
        url = f"{GOVUK_COMPANY_BASE}/{company_number}"
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "lxml")
        else:
            return None
    except Exception as e:
        return None

def extract_registered_office_text(soup: BeautifulSoup) -> str:
    if not soup:
        return ""
    label = soup.find(string=re.compile(r"Registered office address", re.I))
    if not label:
        return ""
    container = label.find_parent()
    if not container:
        return ""
    dd = container.find_next("dd")
    if dd:
        text = dd.get_text(" ", strip=True)
        if ("company type" not in text.lower()) and ("company status" not in text.lower()):
            return text
    for tag in label.find_all_next(["address", "p", "div", "span", "li"], limit=10):
        t = tag.get_text(" ", strip=True)
        if POSTCODE_RE.search(t):
            if ("company type" not in t.lower()) and ("company status" not in t.lower()):
                if "," in t:
                    return t
    fulltext = soup.get_text(" ", strip=True)
    m = POSTCODE_RE.search(fulltext)
    if m:
        start = max(0, m.start() - 120)
        end = min(len(fulltext), m.end() + 60)
        return fulltext[start:end].strip()
    return ""

def extract_sic_full_text(soup: BeautifulSoup) -> str:
    if not soup:
        return ""
    el = soup.find("dd", id=re.compile(r"sic", re.I))
    if el and el.get_text(strip=True):
        return el.get_text(" ", strip=True)
    label = soup.find(string=re.compile(r"Nature of business", re.I))
    if label:
        parent = label.parent
        for candidate in parent.find_all_next(["dd", "li", "p", "span"], limit=6):
            t = candidate.get_text(" ", strip=True)
            if t and len(t) > 5:
                return t
    return ""

# -------------------------
# GROQ FUNCTIONS (Same as your existing code)
# -------------------------
groq_client = None
if GROQ_API_KEY and Groq:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        groq_client = None

def pick_city_with_groq(address_text: str) -> tuple[str, str]:
    if not address_text:
        return "", ""
    
    explicit_city = find_explicit_city_in_address(address_text)
    if explicit_city:
        return explicit_city, "Explicit Match"
    
    sub = pick_city_heuristic_substring(address_text)
    if sub:
        return sub, "Heuristic Substring"
    
    groq_failed_reason = ""

    if groq_client:
        prompt = (
            f"GEOGRAPHICAL CLASSIFICATION - SYSTEMATIC DISTANCE CHECKING\n\n"
            f"Address: {address_text}\n\n"
            f"STEP-BY-STEP PROCESS:\n"
            f"1. Calculate ACTUAL road distances to ALL cities in the list from the address\n"
            f"2. RANK cities from CLOSEST to FURTHEST\n"
            f"3. START from the CLOSEST city and check if it's in the valid list\n"
            f"4. If closest city is NOT in list, move to NEXT CLOSEST\n"
            f"5. Continue until you find the FIRST city that IS in the list\n"
            f"6. Return ONLY that city name\n\n"
            f"VALID CITIES (MUST be from this list):\n{', '.join(CITY_WHITELIST)}\n\n"
            f"RULES:\n"
            f"- You MUST return the CLOSEST city from the valid list\n"
            f"- Check cities in order: closest first, then next closest\n"
            f"- No explanations, just the single city name\n"
            f"- Double-check your geographical calculations\n\n"
            f"Closest valid city:"
        )
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            if hasattr(resp.choices[0], 'message'):
                message = resp.choices[0].message
                if hasattr(message, 'content'):
                    raw_response = message.content.strip()
                elif isinstance(message, dict) and 'content' in message:
                    raw_response = message['content'].strip()
                else:
                    raw_response = str(message).strip()
            else:
                raw_response = str(resp.choices[0]).strip()
            
            candidate = ""
            for city in CITY_WHITELIST:
                if raw_response.lower().strip() == city.lower():
                    candidate = city
                    break
            
            if candidate:
                return candidate, "Groq AI"
            else:
                groq_failed_reason = f"Groq returned '{raw_response}' which is NOT in whitelist"
                
        except Exception as e:
            groq_failed_reason = f"Groq error: {e}"
    
    heuristic_city = pick_city_heuristic(address_text)
    if heuristic_city:
        method = "Heuristic Fallback"
        if groq_failed_reason:
            method += f" ({groq_failed_reason})"
        return heuristic_city, method
    else:
        return "", "No Match Found"

def find_explicit_city_in_address(address_text: str) -> str:
    if not address_text:
        return ""
    address_lower = address_text.lower()
    for city in CITY_WHITELIST:
        city_lower = city.lower()
        if re.search(r'\b' + re.escape(city_lower) + r'\b', address_lower):
            return city
    return ""

def pick_city_heuristic_substring(address_text: str) -> str:
    if not address_text:
        return ""
    text = address_text.lower()
    for city in CITY_WHITELIST:
        if city.lower() in text:
            return city
    return ""

def pick_city_heuristic(address_text: str) -> str:
    if not address_text:
        return ""
    candidate = process.extractOne(address_text, CITY_WHITELIST, scorer=fuzz.partial_ratio)
    if candidate:
        city_name, score, _ = candidate
        if score >= 45:
            return city_name
    return ""

# -------------------------
# STREAMLIT PROCESSING FUNCTION
# -------------------------
def process_companies_dataframe(df, progress_bar, status_text):
    """
    Process the uploaded DataFrame using your existing logic
    """
    # Find business name column
    col_candidates = [c for c in df.columns if c.strip().lower() == "business name" or "business" in c.lower() and "name" in c.lower()]
    if col_candidates:
        business_col = col_candidates[0]
    else:
        business_col = df.columns[0]
    
    # Add output columns
    for out_col in ["Company status", "Address", "City", "Postal code", "Short description", "Classification Method"]:
        if out_col not in df.columns:
            df[out_col] = ""

    total_rows = len(df)
    results = []
    
    for idx in range(total_rows):
        name = str(df.at[idx, business_col]).strip() if pd.notna(df.at[idx, business_col]) else ""
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{total_rows}: {name[:50]}{'...' if len(name) > 50 else ''}")
        
        if not name:
            df.at[idx, "Company status"] = "empty_name"
            results.append(df.iloc[idx].to_dict())
            continue

        # 1) Search Companies House
        items = ch_search_company(name, max_results=5)
        polite_sleep()
        
        if items is None:
            df.at[idx, "Company status"] = "search_error"
            results.append(df.iloc[idx].to_dict())
            continue
            
        if not items:
            df.at[idx, "Company status"] = "no_match"
            results.append(df.iloc[idx].to_dict())
            continue

        # Choose best result
        chosen = None
        for it in items:
            title = it.get("title") or ""
            if title.strip().lower() == name.strip().lower():
                chosen = it
                break
        if not chosen:
            chosen = items[0]

        company_number = chosen.get("company_number")
        
        # 2) Fetch full profile
        profile = ch_get_company_profile(company_number)
        polite_sleep()
        
        if not profile:
            df.at[idx, "Company status"] = "profile_fetch_error"
            results.append(df.iloc[idx].to_dict())
            continue

        # Always write company status
        status = profile.get("company_status", "") or ""
        df.at[idx, "Company status"] = status

        # Get address data
        api_address = profile.get("registered_office_address", {}) or {}
        api_postcode = api_address.get("postal_code") or ""

        # 3) Fetch gov.uk HTML
        govuk_soup = fetch_govuk_soup(company_number)
        gov_raw_address = extract_registered_office_text(govuk_soup)
        
        if not gov_raw_address:
            structured_parts = [
                api_address.get("address_line_1",""),
                api_address.get("address_line_2",""),
                api_address.get("locality",""),
                api_address.get("region",""),
                api_address.get("postal_code",""),
                api_address.get("country","")
            ]
            gov_raw_address = ", ".join([p for p in structured_parts if p])

        # 4) Extract landmark-limited address
        pasted_address = first_landmark_address(gov_raw_address) if gov_raw_address else ""
        if not pasted_address and api_address:
            pasted_address = api_address.get("address_line_1","") or (api_address.get("address_line_2","") or "")

        # 5) Postal code
        postal_code = api_postcode or extract_postcode(gov_raw_address)

        # 6) Nature of business
        short_desc = extract_sic_full_text(govuk_soup)
        if not short_desc:
            sic_codes = profile.get("sic_codes", []) or []
            if sic_codes:
                short_desc = "SIC Code " + sic_codes[0]
            else:
                short_desc = ""

        # 7) City resolution
        city_choice, classification_method = pick_city_with_groq(gov_raw_address or pasted_address or short_desc)
        
        # 8) Update row
        df.at[idx, "Address"] = pasted_address
        df.at[idx, "City"] = city_choice
        df.at[idx, "Postal code"] = postal_code
        df.at[idx, "Short description"] = short_desc
        df.at[idx, "Classification Method"] = classification_method

        results.append(df.iloc[idx].to_dict())
        polite_sleep()

    return pd.DataFrame(results)

# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.markdown('<div class="main-header">🏙️ UK Company City Classifier</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Upload your CSV file containing company names
    
    This tool will:
    - Look up each company in Companies House
    - Extract registered office addresses and business descriptions  
    - Classify the closest UK city using AI
    - Provide a downloadable enriched CSV file
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} companies")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Processing options
            st.subheader("Processing Options")
            sample_size = st.slider(
                "Number of companies to process (use smaller numbers for testing)", 
                min_value=1, 
                max_value=len(df), 
                value=min(10, len(df))
            )
            
            if st.button("🚀 Start Processing", type="primary"):
                if sample_size > 50:
                    st.warning("⚠️ Processing large files may take several minutes. Consider using smaller samples for testing.")
                
                # Create progress elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_placeholder = st.empty()
                
                # Process the data
                sample_df = df.head(sample_size).copy()
                results_df = process_companies_dataframe(sample_df, progress_bar, status_text)
                
                # Update progress to complete
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Show results
                st.subheader("Processing Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", len(results_df))
                with col2:
                    success_count = len(results_df[results_df["Company status"] == "active"])
                    st.metric("Active Companies", success_count)
                with col3:
                    city_count = len(results_df[results_df["City"] != ""])
                    st.metric("Cities Found", city_count)
                with col4:
                    groq_count = len(results_df[results_df["Classification Method"] == "Groq AI"])
                    st.metric("AI Classifications", groq_count)
                
                # Show results table
                st.dataframe(results_df, use_container_width=True)
                
                # Download section
                st.subheader("Download Results")
                
                # Convert DataFrame to CSV for download
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"classified_companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Show classification methods breakdown
                st.subheader("Classification Methods Used")
                method_counts = results_df["Classification Method"].value_counts()
                st.bar_chart(method_counts)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Instructions sidebar
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload CSV** with company names
        2. **Adjust sample size** for testing
        3. **Click Process** to start analysis
        4. **Download results** when complete
        
        ### 📊 Expected Columns
        Your output will include:
        - Company status
        - Cleaned address  
        - Classified city
        - Postal code
        - Business description
        - Classification method
        
        ### ⚡ Processing Notes
        - Uses Companies House API
        - AI-powered city classification
        - Progress tracking
        - Error handling
        """)
        
        st.markdown("### 🔑 API Status")
        if groq_client:
            st.success("✅ Groq API: Connected")
        else:
            st.warning("⚠️ Groq API: Not available")
        
        if COMPANIES_HOUSE_KEY:
            st.success("✅ Companies House API: Connected")
        else:
            st.error("❌ Companies House API: Missing key")

if __name__ == "__main__":
    main()
