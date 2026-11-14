{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% set public_methods = [] %}
   {% for item in methods if not item.startswith('_') %}
     {% set _ = public_methods.append(item) %}
   {% endfor %}
   {% if public_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set public_attributes = [] %}
   {% for item in attributes if not item.startswith('_') %}
     {% set _ = public_attributes.append(item) %}
   {% endfor %}
   {% if public_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in public_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
