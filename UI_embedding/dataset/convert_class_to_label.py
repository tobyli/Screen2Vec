import json
import os

def convert_class_to_text_label(the_class_of_text):
    lookup_class_label_dict = {"AdView": 1, "HtmlBannerWebView":1, "AdContainer":1, "ActionBarOverlayLayout": 2, "TabLayout": 2, "TabLayout$SlidingTabStrip": 2,  "TabLayout$TabView": 2, "LinearLayout": 2, "FitWindowsLinearLayout": 2, "CustomScreenLinearLayout": 2, "AppBarLayout": 2, "FrameLayout": 2, "ContentFrameLayout": 2, "FitWindowsFrameLayout": 2, "NoSaveStateFrameLayout": 2, "RelativeLayout": 2, "TableLayout": 2, "BottomTagGroupView": 3, "BottomBar": 3, "ButtonBar": 4, "CardView": 5, "CheckBox": 6, "CheckedTextView":6, "DrawerLayout": 7, "DatePicker": 8, "ImageView": 9, "ImageButton": 10, "GlyphView": 10, "AppCompactButton": 10, "AppCompactImageButton": 10, "ActionMenuItemView":10, "ActionMenuItemPresenter":10, "EditText": 11, "SearchBoxView": 11, "AppCompatAutoCompleteTextView": 11, "TextView": 11, "TextInputLayout": 11,  "ListView": 12, "RecyclerView": 12, "ListPopUpWindow": 12, "tabItem": 12, "GridView": 12, "MapView": 13, "SlidingTab": 14, "NumberPicker": 15, "Switch": 16, "SwitchCompat":16, "ViewPageIndicatorDots": 17, "PageIndicator": 17, "CircleIndicator": 17, "PagerIndicator": 17, "RadioButton": 18, "CheckedTextView": 18, "SeekBar": 19, "Button": 20, "TextView": 20, "ToolBar": 21, "Toolbar": 21, "TitleBar": 21, "ActionBar": 21, "VideoView": 22, "WebView": 23}
    #lookup_class_label_dict = {"AdView": "Advertisement", "HtmlBannerWebView":"Advertisement", "AdContainer":"Advertisement", "ImageView":"Background Image", "BottomTagGroupView": "Bottom Navigation", "BottomBar": "Bottom Navigation", "ButtonBar": "Button Bar", "CardView": "Card", "CheckBox": "Checkbox", "CheckedTextView":"Checkbox", "DrawerLayout": "Drawer (Parent)", "DatePicker": "Date Picker", "ImageView": "Image", "ImageButton": "Image Button", "GlyphView": "Image Button", "AppCompactButton": "Image Button", "AppCompactImageButton": "Image Button", "ActionMenuItemView":"Image Button", "ActionMenuItemPresenter":"Image Button", "EditText": "Input", "SearchBoxView": "Input", "AppCompatAutoCompleteTextView": "Input", "TextView": "Input", "ListView": "List Item (Parent)", "RecyclerView": "List Item (Parent)", "ListPopUpWindow": "List Item (Parent)", "tabItem": "List Item (Parent)", "GridView": "List Item (Parent)", "MapView": "Map View", "SlidingTab": "Multi-Tab", "NumberPicker": "Number Stepper", "Switch": "On/Off Switch", "ViewPageIndicatorDots": "Pager Indicator", "PageIndicator": "Pager Indicator", "CircleIndicator": "Pager Indicator", "PagerIndicator": "Pager Indicator", "RadioButton": "Radio Button", "CheckedTextView": "Radio Button", "SeekBar": "Slider", "Button": "Text Button", "TextView": "Text Button", "ToolBar": "Toolbar", "TitleBar": "Toolbar", "ActionBar": "Toolbar", "VideoView": "Video", "WebView": "Web View"}
    the_class_of_text = the_class_of_text.split(".")[-1]
    if lookup_class_label_dict.get(the_class_of_text):
        return lookup_class_label_dict.get(the_class_of_text)
    else: 
        for key, val in lookup_class_label_dict.items():
            if the_class_of_text.endswith(key):
                return val
            if the_class_of_text.startswith(key):
                return val
        return 0
        #return 'Unrecognized Feature'
