import json
import os

def convert_class_to_text_label(the_class_of_text):
    lookup_class_label_dict = {"AdView": "Advertisement", "HtmlBannerWebView":"Advertisement", "AdContainer":"Advertisement", "ImageView":"Background Image", "BottomTagGroupView": "Bottom Navigation", "BottomBar": "Bottom Navigation", "ButtonBar": "Button Bar", "CardView": "Card", "CheckBox": "Checkbox", "CheckedTextView":"Checkbox", "DrawerLayout": "Drawer (Parent)", "DatePicker": "Date Picker", "ImageView": "Image", "ImageButton": "Image Button", "GlyphView": "Image Button", "AppCompactButton": "Image Button", "AppCompactImageButton": "Image Button", "ActionMenuItemView":"Image Button", "ActionMenuItemPresenter":"Image Button", "EditText": "Input", "SearchBoxView": "Input", "AppCompatAutoCompleteTextView": "Input", "TextView": "Input", "ListView": "List Item (Parent)", "RecyclerView": "List Item (Parent)", "ListPopUpWindow": "List Item (Parent)", "tabItem": "List Item (Parent)", "GridView": "List Item (Parent)", "MapView": "Map View", "SlidingTab": "Multi-Tab", "NumberPicker": "Number Stepper", "Switch": "On/Off Switch", "ViewPageIndicatorDots": "Pager Indicator", "PageIndicator": "Pager Indicator", "CircleIndicator": "Pager Indicator", "PagerIndicator": "Pager Indicator", "RadioButton": "Radio Button", "CheckedTextView": "Radio Button", "SeekBar": "Slider", "Button": "Text Button", "TextView": "Text Button", "ToolBar": "Toolbar", "TitleBar": "Toolbar", "ActionBar": "Toolbar", "VideoView": "Video", "WebView": "Web View"}
    the_class_of_text = the_class_of_text.split(".")[-1]
    if lookup_class_label_dict.get(the_class_of_text):
        return lookup_class_label_dict.get(the_class_of_text)
    else: 
        for key, val in lookup_class_label_dict.items():
            if the_class_of_text.endswith(key):
                return val
        return 'Unrecognized Feature'
