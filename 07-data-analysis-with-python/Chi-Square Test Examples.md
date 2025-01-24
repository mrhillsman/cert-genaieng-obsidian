## Practical Example - Weak Correlation

Suppose a researcher wants to determine if there is an association between gender (male, female) and preference for a new product (like, dislike). The researcher surveys 100 people and records the following data:

| Category | Like | Dislike | Total |
|----------|------|---------|-------|
| Male     | 20   | 30      | 50    |
| Female   | 25   | 25      | 50    |
| Total    | 45   | 55      | 100   |

---

### Step 1: Calculate Expected Frequencies

Using the formula for expected frequencies:
$$
\Large{E_{Male,Like} = \frac{50 \times 45}{100} = 22.5}
$$

$$
\Large{E_{Male,Dislike} = \frac{50 \times 55}{100} = 27.5}
$$

$$
\Large{E_{Female,Like} = \frac{50 \times 45}{100} = 22.5}
$$

$$
\Large{E_{Female,Dislike} = \frac{50 \times 55}{100} = 27.5}
$$
---

### Step 2: Compute Chi-Square Statistic

The chi-square statistic is calculated using the formula:
$$
\Large{\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}}
$$

For this example:
$$
\Large{\chi^2 = \frac{(20 - 22.5)^2}{22.5} + \frac{(30 - 27.5)^2}{27.5} + \frac{(25 - 22.5)^2}{22.5} + \frac{(25 - 27.5)^2}{27.5}}
$$

Breaking it down:
$$
\Large{\chi^2 = \frac{(2.5)^2}{22.5} + \frac{(2.5)^2}{27.5} + \frac{(2.5)^2}{22.5} + \frac{(2.5)^2}{27.5}}
$$

Simplify:
$$
\Large{\chi^2 = \frac{6.25}{22.5} + \frac{6.25}{27.5} + \frac{6.25}{22.5} + \frac{6.25}{27.5}}
$$
$$
\Large{\chi^2 = 0.277 + 0.227 + 0.277 + 0.227}
$$
$$
\Large{\chi^2 = 1.008}
$$
---

### Step 3: Determine Degrees of Freedom

The formula for degrees of freedom is:
$$
\Large{df = (r - 1) \times (c - 1)}
$$

Where \( r \) is the number of rows and \( c \) is the number of columns. For this example:
$$
\Large{df = (2 - 1) \times (2 - 1) = 1}
$$
---

### Step 4: Interpret the Result

Using a chi-square distribution table, we compare the calculated chi-square value (1.008) with the critical value at one degree of freedom and a significance level (e.g., 0.05). The critical value, as determined from chi-square distribution tables, is approximately 3.841.

Since (1.008 < 3.841), we fail to reject the null hypothesis. Thus, there is no significant association between gender and product preference in this sample.

---

## Practical Example - Strong Association

Consider a study investigating the relationship between smoking status (smoker, non-smoker) and the incidence of lung disease (disease, no disease). The researcher collects data from 200 individuals and records the following information:

| Category   | Disease | No Disease | Total |
|------------|---------|------------|-------|
| Smoker     | 50      | 30         | 80    |
| Non-Smoker | 20      | 100        | 120   |
| Total      | 70      | 130        | 200   |

---

### Step 1: Calculate Expected Frequencies

Using the formula for expected frequencies:
$$
\Large{E_{Smoker,Disease} = \frac{80 \times 70}{200} = 28}
$$

$$
\Large{E_{Smoker,No\ Disease} = \frac{80 \times 130}{200} = 52}
$$

$$
\Large{E_{Non-Smoker,Disease} = \frac{120 \times 70}{200} = 42}
$$

$$
\Large{E_{Non-Smoker,No\ Disease} = \frac{120 \times 130}{200} = 78}
$$
---

### Step 2: Compute Chi-Square Statistic

The chi-square statistic is calculated using the formula:
$$
\Large{\chi^2 = \sum \frac{(O - E)^2}{E}}
$$

For this example:
$$
\Large{\chi^2 = \frac{(50 - 28)^2}{28} + \frac{(30 - 52)^2}{52} + \frac{(20 - 42)^2}{42} + \frac{(100 - 78)^2}{78}}
$$

Breaking it down:
$$
\Large{\chi^2 = \frac{(22)^2}{28} + \frac{(22)^2}{52} + \frac{(22)^2}{42} + \frac{(22)^2}{78}}
$$

Simplify:
$$
\Large{\chi^2 = \frac{484}{28} + \frac{484}{52} + \frac{484}{42} + \frac{484}{78}}
$$
$$
\Large{\chi^2 = 17.29 + 9.31 + 11.52 + 6.21}
$$
$$
\Large{\chi^2 = 44.33}
$$
---

### Step 3: Determine Degrees of Freedom

The formula for degrees of freedom is:
$$
\Large{df = (r - 1) \times (c - 1)}
$$

Where \( r \) is the number of rows and \( c \) is the number of columns. For this example:
$$
\Large{df = (2 - 1) \times (2 - 1) = 1}
$$
---

### Step 4: Interpret the Result

Using a chi-square distribution table, we compare the calculated chi-square value (44.33) with the critical value at one degree of freedom and a significance level (e.g., 0.05), approximately 3.841. 

Since (44.33 > 3.841), we reject the null hypothesis. This indicates a significant association between smoking status and the incidence of lung disease in this sample.
