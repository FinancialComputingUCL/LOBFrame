class Trading:
    def __init__(self, trading_hyperparameters):
        self.long_inventory = 0
        self.short_inventory = 0
        self.long_price = 0
        self.short_price = 0
        self.date_time_entry_long = None
        self.date_time_exit_long = None
        self.date_time_entry_short = None
        self.date_time_exit_short = None
        self.trading_history = []

    def long(self, price, datetime=None):
        amount = 1
        self.long_inventory += amount
        self.long_price = price
        self.date_time_entry_long = datetime

    def short(self, price, datetime=None):
        amount = 1
        self.short_inventory += amount
        self.short_price = price
        self.date_time_entry_short = datetime

    def exit_long(self, price, datetime=None):
        self.trading_history.append({'Type': 'Long', 'Entry_Long': self.date_time_entry_long, 'Price_Entry_Long': self.long_price,
                                     'Exit_Long': datetime, 'Price_Exit_Long': price})

        self.long_inventory = 0
        self.long_price = 0
        self.date_time_entry_long = None

    def exit_short(self, price, datetime=None):
        self.trading_history.append({'Type': 'Short', 'Entry_Short': self.date_time_entry_short, 'Price_Entry_Short': self.short_price,
                                     'Exit_Short': datetime, 'Price_Exit_Short': price})

        self.short_inventory = 0
        self.short_price = 0
        self.date_time_entry_short = None
